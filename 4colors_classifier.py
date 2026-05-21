"""
4-Color + Idle EEG Classifier System
Classifies brain responses to Red, Blue, Green, Yellow visual stimuli,
imagination of colors, and an Idle (no stimulus) state.

ML methodology matched to red_blue_classifier.py.
"""

import time
import random
import numpy as np
import logging
import pandas as pd
import joblib
import os
import argparse
import sys
from datetime import datetime
from typing import Tuple, Dict, List, Optional
from threading import Event
from collections import deque

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations, DetrendOperations, FilterTypes, NoiseTypes
from pyqtgraph.Qt import QtWidgets, QtGui, QtCore

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ==============================
# CONFIGURATION
# ==============================
class Config:
    # Hardware
    SERIAL_PORT = "/dev/cu.usbserial-DP04VYIJ"
    BOARD_ID = BoardIds.CYTON_BOARD.value

    # Data Collection
    TRIAL_DURATION = 3.0        # seconds per stimulus
    REST_DURATION = 1.0         # seconds between trials
    TRIALS_PER_COLOR = 5  # trials per color class
    TRIALS_PER_IDLE = 10        # trials for idle class
    PREPARATION_TIME = 1.5      # fixation cross before stimulus
    POST_STIMULUS_BUFFER = 1  # buffer after stimulus before processing
    IMAGINATION_DURATION = 3.0  # seconds for imagination phase

    # Data Processing
    SAMPLING_RATE = 250         # Cyton board sampling rate
    IMPEDANCE_THRESHOLD = 50000 # microVolts
    FILTER_LOW = 0.5            # Hz - high-pass filter
    FILTER_HIGH = 40.0          # Hz - low-pass filter
    NOTCH_FREQ = 60.0           # Hz - US powerline (50 for EU)

    # Feature Extraction — matched to red_blue_classifier (18 per channel)
    WINDOW_SIZE = 2.0           # seconds for feature extraction windows
    OVERLAP = 0.5               # window overlap (50%)

    # Directories (kept separate from red_blue data)
    DATA_DIR = "4colors_data"
    MODEL_DIR = "4colors_models"
    RAW_DATA_DIR = "4colors_raw_data"
    PROCESSED_DATA_DIR = "4colors_processed_data"
    IMPEDANCE_LOG = "4colors_impedance_log.csv"

    # Classes: 4 colors + idle
    # NOTE: 'idle' is label 4 — collected during no-stimulus periods
    COLORS = {
        'red':    {'label': 0, 'rgb': (255,   0,   0)},
        'blue':   {'label': 1, 'rgb': (  0,   0, 255)},
        'green':  {'label': 2, 'rgb': (  0, 255,   0)},
        'yellow': {'label': 3, 'rgb': (255, 255,   0)},
        'idle':   {'label': 4, 'rgb': ( 30,  30,  30)},  # dark gray screen
    }
    COLOR_NAMES = ['red', 'blue', 'green', 'yellow', 'idle']

    # Trial types
    TRIAL_TYPES = ['visual', 'imagination']  # both collected per color

    # Training — matched to red_blue_classifier
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    RANDOM_STATE = 42

    # Real-time prediction smoothing window (matched to red_blue vote window)
    PREDICTION_SMOOTHING = 10


# ==============================
# DIRECTORY SETUP
# ==============================
def setup_directories():
    """Create all necessary directories."""
    for d in [Config.DATA_DIR, Config.MODEL_DIR,
              Config.RAW_DATA_DIR, Config.PROCESSED_DATA_DIR]:
        os.makedirs(d, exist_ok=True)


# ==============================
# AUDIO FEEDBACK UTILITY
# ==============================
class AudioFeedback:
    @staticmethod
    def play_beep():
        """Cross-platform beep."""
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(1000, 300)
                return
        except ImportError:
            pass
        try:
            if sys.platform == "darwin":
                os.system('afplay /System/Library/Sounds/Pop.aiff 2>/dev/null &')
                return
        except Exception:
            pass
        try:
            if sys.platform.startswith("linux"):
                os.system('speaker-test -t sine -f 1000 -l 1 -s 1 2>/dev/null &')
                return
        except Exception:
            pass
        print('\a', end='', flush=True)

    @staticmethod
    def play_completion_sound():
        try:
            if sys.platform == "darwin":
                os.system('afplay /System/Library/Sounds/Glass.aiff 2>/dev/null &')
                return
        except Exception:
            pass
        print('\a', end='', flush=True)


# ==============================
# SYNCHRONIZATION MANAGER
# ==============================
class SyncManager:
    def __init__(self):
        self.data_ready = Event()
        self.processing_complete = Event()
        self.collection_active = Event()

    def start_collection(self):
        self.collection_active.set()
        self.data_ready.clear()
        self.processing_complete.clear()

    def signal_data_ready(self):
        self.data_ready.set()

    def wait_for_data(self, timeout: float = 10.0) -> bool:
        return self.data_ready.wait(timeout)

    def signal_processing_complete(self):
        self.processing_complete.set()

    def wait_for_processing(self, timeout: float = 10.0) -> bool:
        return self.processing_complete.wait(timeout)

    def stop_collection(self):
        self.collection_active.clear()

    def reset(self):
        self.data_ready.clear()
        self.processing_complete.clear()


# ==============================
# IMPEDANCE MONITORING
# ==============================
class ImpedanceMonitor:
    def __init__(self, board: BoardShim, threshold: float = Config.IMPEDANCE_THRESHOLD):
        self.board = board
        self.threshold = threshold
        self.impedance_log = []

    def check_impedance(self) -> Dict[int, float]:
        eeg_channels = BoardShim.get_eeg_channels(self.board.get_board_id())
        impedances = {}
        try:
            data = self.board.get_current_board_data(Config.SAMPLING_RATE)
        except Exception:
            for ch in eeg_channels:
                impedances[ch] = float('inf')
            return impedances
        for ch in eeg_channels:
            if data.shape[1] > 0:
                impedances[ch] = np.var(data[ch]) * 1000
            else:
                impedances[ch] = float('inf')
        self.log_impedance(impedances)
        return impedances

    def log_impedance(self, impedances: Dict[int, float]):
        timestamp = datetime.now().isoformat()
        for ch, imp in impedances.items():
            self.impedance_log.append({
                'timestamp': timestamp, 'channel': ch, 'impedance': imp,
                'status': 'good' if imp < self.threshold else 'poor'
            })

    def get_good_channels(self) -> List[int]:
        impedances = self.check_impedance()
        good = [ch for ch, imp in impedances.items() if imp < self.threshold]
        if not good:
            print("WARNING: No channels with good impedance. Using all channels.")
            good = list(impedances.keys())
        return good

    def save_impedance_log(self):
        setup_directories()
        if self.impedance_log:
            df = pd.DataFrame(self.impedance_log)
            filepath = os.path.join(Config.DATA_DIR, Config.IMPEDANCE_LOG)
            df.to_csv(filepath, index=False)
            print(f"Impedance log saved to {filepath}")


# ==============================
# STIMULUS WINDOW
# ==============================
class FourColorStimulus(QtWidgets.QWidget):
    """
    Full-screen stimulus display.

    Supports three modes per trial:
      1. Visual   — show the actual color
      2. Imagine  — show a dark screen with a text prompt to imagine the color
      3. Idle     — show a near-black screen with no instruction
    """

    def __init__(self):
        super().__init__() #FIX THIS
        self.setWindowTitle("4-Color + Idle Visual Stimulus")

        # Normal resizable window
        self.setWindowFlags(QtCore.Qt.Window)

        # Optional: keep on top
        # self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        # Set window size
        self.resize(1000, 700)

        # Optional: position on screen
        self.move(100, 100)

        # Show normally instead of fullscreen
        self.show()
        self.audio = AudioFeedback()
        self._current_color = None
        self._stimulus_start_time = None

        # --- Instruction label (centered, large) ---
        self.instruction_label = QtWidgets.QLabel(self)
        self.instruction_label.setAlignment(QtCore.Qt.AlignCenter)
        self.instruction_label.setStyleSheet("""
            QLabel {
                font-size: 32px;
                font-weight: bold;
                color: white;
                background-color: rgba(0, 0, 0, 160);
                padding: 30px;
                border-radius: 15px;
            }
        """)
        self.instruction_label.hide()

        # --- Progress label (top center) ---
        self.progress_label = QtWidgets.QLabel(self)
        self.progress_label.setAlignment(QtCore.Qt.AlignCenter)
        self.progress_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                color: white;
                background-color: rgba(0, 0, 0, 100);
                padding: 10px;
                border-radius: 5px;
            }
        """)
        self.progress_label.hide()

        # --- Countdown label ---
        self.timer_label = QtWidgets.QLabel(self)
        self.timer_label.setAlignment(QtCore.Qt.AlignCenter)
        self.timer_label.setStyleSheet("""
            QLabel {
                font-size: 52px;
                font-weight: bold;
                color: white;
                background-color: rgba(0, 0, 0, 100);
                padding: 15px;
                border-radius: 10px;
            }
        """)
        self.timer_label.hide()

        # --- Phase label (VISUAL / IMAGINE / IDLE, below center) ---
        self.phase_label = QtWidgets.QLabel(self)
        self.phase_label.setAlignment(QtCore.Qt.AlignCenter)
        self.phase_label.setStyleSheet("""
            QLabel {
                font-size: 22px;
                font-style: italic;
                color: rgba(255,255,255,200);
                background-color: rgba(0,0,0,80);
                padding: 8px 20px;
                border-radius: 8px;
            }
        """)
        self.phase_label.hide()

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _force_update(self):
        self.repaint()
        QtWidgets.QApplication.processEvents(
            QtCore.QEventLoop.AllEvents,
            1
        )

    # ------------------------------------------------------------------
    # Background color
    # ------------------------------------------------------------------
    def _set_background(self, rgb: Tuple[int, int, int]):
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QtGui.QColor(*rgb))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def show_instruction(self, text: str):
        self.instruction_label.setText(text)
        self.instruction_label.show()
        self.timer_label.hide()
        self._force_update()

    def hide_instruction(self):
        self.instruction_label.hide()
        self._force_update()

    def show_progress(self, current: int, total: int, label: str = ""):
        text = f"Trial {current}/{total}"
        if label:
            text += f"  —  {label.upper()}"
        self.progress_label.setText(text)
        self.progress_label.show()
        self._force_update()

    def hide_progress(self):
        self.progress_label.hide()
        self._force_update()

    def show_countdown(self, seconds: int):
        self.timer_label.setText(str(seconds))
        self.timer_label.show()
        self._force_update()

    def hide_countdown(self):
        self.timer_label.hide()
        self._force_update()

    def show_phase(self, text: str):
        self.phase_label.setText(text)
        self.phase_label.show()
        self._force_update()

    def hide_phase(self):
        self.phase_label.hide()
        # =========================
        # LABEL SIZES
        # =========================
        self.instruction_label.setMinimumWidth(700)
        self.instruction_label.setMaximumWidth(1200)
        self.instruction_label.setWordWrap(True)
        self.progress_label.setFixedWidth(400)
        self.timer_label.setFixedWidth(200)
        self.phase_label.setFixedWidth(500)

        # =========================
        # MAIN LAYOUT
        # =========================
        main_layout = QtWidgets.QVBoxLayout()

        main_layout.setContentsMargins(60, 40, 60, 40)
        main_layout.setSpacing(20)

        # Top progress label
        main_layout.addWidget(
            self.progress_label,
            alignment=QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter
        )

        # Spacer
        main_layout.addStretch()

        # Main instruction
        main_layout.addWidget(
            self.instruction_label,
            alignment=QtCore.Qt.AlignCenter
        )

        # Countdown timer
        main_layout.addWidget(
            self.timer_label,
            alignment=QtCore.Qt.AlignHCenter
        )

        # Phase label
        main_layout.addWidget(
            self.phase_label,
            alignment=QtCore.Qt.AlignHCenter
        )

        # Bottom spacer
        main_layout.addStretch()

        self.setLayout(main_layout)
        self._force_update()

    def show_fixation_cross(self):
        """Gray screen + crosshair during preparation."""
        self._set_background((64, 64, 64))
        self.hide_countdown()
        self.instruction_label.setText("+")
        self.instruction_label.show()
        self._force_update()

    # ------------------------------------------------------------------
    # Trial display modes
    # ------------------------------------------------------------------
    def show_visual_stimulus(self, color: str) -> float:
        """
        VISUAL mode: fill screen with the actual color.
        Returns precise stimulus onset time.
        """
        self.hide_instruction()
        self.hide_countdown()
        if color.lower() in Config.COLORS:
            self._set_background(Config.COLORS[color.lower()]['rgb'])
        else:
            self._set_background((128, 128, 128))
        self.show_phase("[ VISUAL ]")
        self._current_color = color
        self._force_update()

        # timestamp AFTER screen update
        self._stimulus_start_time = time.perf_counter()

        return self._stimulus_start_time

    def show_imagination_stimulus(self, color: str) -> float:
        """
        IMAGINATION mode: dark screen, text asks participant to vividly
        imagine the color. No actual color is shown.
        Returns precise onset time.
        """
        self.hide_instruction()
        self.hide_countdown()
        # Near-black background — same as idle but with a prompt
        self._set_background((20, 20, 20))
        color_upper = color.upper()
        self.instruction_label.setText(
            f"Vividly\n"
            f"imagine the color  {color_upper}\n\n"
            f"Focus only on  {color_upper}"
        )
        self.instruction_label.show()
        self.show_phase("[ IMAGINE ]")
        self._current_color = color
        self._force_update()

        # timestamp AFTER screen update
        self._stimulus_start_time = time.perf_counter()

        return self._stimulus_start_time

    def show_idle_stimulus(self) -> float:
        """
        IDLE mode: dark screen, no instruction — participant rests freely.
        Returns precise onset time.
        """
        self.hide_instruction()
        self.hide_countdown()
        self._set_background(Config.COLORS['idle']['rgb'])
        self.show_phase("[ REST — let your mind wander ]")
        self._current_color = 'idle'
        self._force_update()

        # timestamp AFTER screen update
        self._stimulus_start_time = time.perf_counter()

        return self._stimulus_start_time

    def show_rest_screen(self):
        """Black rest screen between trials."""
        self.hide_instruction()
        self.hide_phase()
        self._set_background((0, 0, 0))
        self._force_update()

    def play_beep(self):
        self.audio.play_beep()

    def play_completion(self):
        self.audio.play_completion_sound()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()


# ==============================
# FEATURE EXTRACTION
# Matched to red_blue_classifier: 18 features per channel
# ==============================
class AdvancedFeatureExtractor:
    """
    Extracts exactly 18 features per EEG channel, matching the
    red_blue_classifier feature set for direct model compatibility.
    """

    FEATURES_PER_CHANNEL = 18

    def __init__(self, sampling_rate: int = Config.SAMPLING_RATE):
        self.sampling_rate = sampling_rate
        self.window_size = int(Config.WINDOW_SIZE * sampling_rate)

    def preprocess_data(self, data: np.ndarray, channels: List[int]) -> np.ndarray:
        """Band-pass + notch + environmental noise removal, per channel."""
        processed = data.copy()
        for ch in channels:
            if ch >= processed.shape[0]:
                continue
            ch_data = processed[ch]
            if len(ch_data) == 0:
                continue
            try:
                DataFilter.detrend(ch_data, DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(
                    ch_data, self.sampling_rate,
                    Config.FILTER_LOW, Config.FILTER_HIGH, 4,
                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
                )
                DataFilter.perform_bandstop(
                    ch_data, self.sampling_rate,
                    Config.NOTCH_FREQ - 2, Config.NOTCH_FREQ + 2, 4,
                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
                )
                DataFilter.remove_environmental_noise(
                    ch_data, self.sampling_rate, NoiseTypes.SIXTY.value
                )
            except Exception as e:
                print(f"  Warning: preprocess channel {ch}: {e}")
        return processed

    def extract_comprehensive_features(self, data: np.ndarray, channels: List[int]) -> np.ndarray:
        """
        Extract 18 features per channel:
          Time domain  (6): mean, std, peak-to-peak, var, median, IQR
          Freq bands   (5): relative delta, theta, alpha, beta, gamma power
          Band ratios  (3): alpha/beta, theta/alpha, beta/gamma
          Visual bands (2): occipital alpha, visual gamma (relative)
          Extra ratios (2): low_beta/high_beta, gamma/alpha
                            (padding zeros to match red_blue's 18 count)
        Total = 6 + 5 + 3 + 2 + 2 = 18  ✓
        """
        features = []
        for ch in channels:
            if ch >= data.shape[0]:
                features.extend([0.0] * self.FEATURES_PER_CHANNEL)
                continue
            ch_data = data[ch]
            if len(ch_data) == 0:
                features.extend([0.0] * self.FEATURES_PER_CHANNEL)
                continue
            try:
                # --- Time domain (6) ---
                features.extend([
                    np.mean(ch_data),
                    np.std(ch_data),
                    np.max(ch_data) - np.min(ch_data),
                    np.var(ch_data),
                    np.median(ch_data),
                    np.percentile(ch_data, 75) - np.percentile(ch_data, 25),
                ])

                # --- Frequency domain (12) ---
                try:
                    psd = DataFilter.get_psd_welch(
                        ch_data,
                        DataFilter.get_nearest_power_of_two(self.sampling_rate),
                        self.sampling_rate // 2,
                        self.sampling_rate,
                        WindowOperations.BLACKMAN_HARRIS.value
                    )
                    delta  = DataFilter.get_band_power(psd, 0.5,  4.0)
                    theta  = DataFilter.get_band_power(psd, 4.0,  8.0)
                    alpha  = DataFilter.get_band_power(psd, 8.0, 12.0)
                    beta   = DataFilter.get_band_power(psd, 12.0, 30.0)
                    gamma  = DataFilter.get_band_power(psd, 30.0, 40.0)
                    total  = max(delta + theta + alpha + beta + gamma, 1e-10)

                    # Relative band powers (5)
                    features.extend([
                        delta / total,
                        theta / total,
                        alpha / total,
                        beta  / total,
                        gamma / total,
                    ])
                    # Band ratios (3)
                    features.extend([
                        alpha / max(beta,  1e-10),
                        theta / max(alpha, 1e-10),
                        beta  / max(gamma, 1e-10),
                    ])
                    # Visual-specific (2)  — occipital alpha, visual gamma
                    occ_alpha   = DataFilter.get_band_power(psd, 8.0,  13.0)
                    vis_gamma   = DataFilter.get_band_power(psd, 30.0, 45.0)
                    features.extend([
                        occ_alpha / total,
                        vis_gamma / total,
                    ])
                    # Extra discriminative ratios (2)
                    low_beta  = DataFilter.get_band_power(psd, 12.0, 20.0)
                    high_beta = DataFilter.get_band_power(psd, 20.0, 30.0)
                    features.extend([
                        low_beta / max(high_beta, 1e-10),
                        gamma    / max(alpha,     1e-10),
                    ])
                    # total freq features = 5+3+2+2 = 12 ✓

                except Exception as e:
                    print(f"  Warning: PSD ch {ch}: {e}")
                    features.extend([0.0] * 12)

            except Exception as e:
                print(f"  Warning: feature ch {ch}: {e}")
                features.extend([0.0] * self.FEATURES_PER_CHANNEL)

        return np.array(features)


# ==============================
# RAW DATA SAVER
# ==============================
class RawDataSaver:
    def __init__(self, subject_id: str, session_name: str):
        self.subject_id = subject_id
        self.session_name = session_name
        self.trial_count = 0
        setup_directories()
        self.session_dir = os.path.join(
            Config.RAW_DATA_DIR,
            f"{subject_id}_{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.session_dir, exist_ok=True)

    def save_trial(self, data: np.ndarray, color: str, label: int,
                   trial_type: str, channels: List[int], metadata: Dict) -> str:
        self.trial_count += 1
        fname = f"trial_{self.trial_count:03d}_{color}_{trial_type}_label{label}.npz"
        fpath = os.path.join(self.session_dir, fname)
        np.savez(fpath, data=data, color=color, label=label,
                 trial_type=trial_type, channels=channels,
                 sampling_rate=Config.SAMPLING_RATE, **metadata)
        return fpath


# ==============================
# DATA COLLECTION MANAGER
# ==============================
class DataCollectionManager:
    """
    Collects EEG data across three trial types:
      • Visual     — participant watches the colored screen
      • Imagination — participant imagines the color (dark screen)
      • Idle        — participant rests with no instruction
    """

    def __init__(self, board: BoardShim, subject_id: str):
        self.board = board
        self.subject_id = subject_id
        self.impedance_monitor = ImpedanceMonitor(board)
        self.feature_extractor = AdvancedFeatureExtractor()
        self.sync_manager = SyncManager()
        setup_directories()

    # ------------------------------------------------------------------
    # Timing helpers
    # ------------------------------------------------------------------
    def _wait_precise(self, duration: float):
        end_time = time.perf_counter() + duration

        while time.perf_counter() < end_time:
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.AllEvents,
                1
            )
            time.sleep(0.001)

    def _collect_trial_data_sync(self, duration: float) -> np.ndarray:
        num_samples = int(duration * Config.SAMPLING_RATE)
        self.board.get_board_data()   # flush stale data
        start = time.perf_counter()
        self._wait_precise(duration)
        elapsed = time.perf_counter() - start
        self._wait_precise(Config.POST_STIMULUS_BUFFER)
        data = self.board.get_board_data()
        print(f"  Collected: {elapsed:.3f}s  |  Samples: {data.shape[1]}")
        return data

    # ------------------------------------------------------------------
    # Session builder
    # ------------------------------------------------------------------
    def _build_trial_list(self) -> List[Dict]:
        """
        Build a balanced, shuffled trial list.
        Each color gets TRIALS_PER_COLOR visual + TRIALS_PER_COLOR imagination trials.
        Idle gets TRIALS_PER_IDLE visual (actual rest) trials.
        """
        trials = []
        for color_name, color_info in Config.COLORS.items():
            if color_name == 'idle':
                for _ in range(Config.TRIALS_PER_IDLE):
                    trials.append({
                        'color': 'idle',
                        'label': color_info['label'],
                        'trial_type': 'idle',
                    })
            else:
                for _ in range(Config.TRIALS_PER_COLOR):
                    trials.append({
                        'color': color_name,
                        'label': color_info['label'],
                        'trial_type': 'visual',
                    })
                for _ in range(Config.TRIALS_PER_COLOR):
                    trials.append({
                        'color': color_name,
                        'label': color_info['label'],
                        'trial_type': 'imagination',
                    })
        random.shuffle(trials)
        return trials

    # ------------------------------------------------------------------
    # Main collection loop
    # ------------------------------------------------------------------
    def collect_session_data(self, session_name: str = None) -> Optional[pd.DataFrame]:
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"\n{'='*60}")
        print(f"  4-Color + Idle Data Collection  |  {session_name}")
        print(f"{'='*60}")

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        stimulus = FourColorStimulus()
        raw_saver = RawDataSaver(self.subject_id, session_name)

        try:
            # Impedance check
            print("\nChecking electrode impedance...")
            good_channels = self.impedance_monitor.get_good_channels()
            print(f"Good channels: {good_channels}")
            if len(good_channels) < 2:
                stimulus.show_instruction("⚠  Check electrode connections\n\nContinuing in 5 s…")
                self._wait_precise(5.0)

            # Start stream
            self.board.start_stream()
            print("Stream started — stabilising…")
            self._wait_precise(2.0)
            self.board.get_board_data()   # flush

            trials = self._build_trial_list()
            total = len(trials)
            n_visual = sum(1 for t in trials if t['trial_type'] == 'visual' and t['color'] != 'idle')
            n_imagine = sum(1 for t in trials if t['trial_type'] == 'imagination')
            n_idle    = sum(1 for t in trials if t['trial_type'] == 'idle')

            print(f"\nTotal trials : {total}")
            print(f"  Visual     : {n_visual}")
            print(f"  Imagination: {n_imagine}")
            print(f"  Idle       : {n_idle}")

            stimulus.show_instruction(
                f"4-Color + Idle Classification\n\n"
                f"{total} trials  "
                f"({n_visual} visual · {n_imagine} imagine · {n_idle} idle)\n\n"
                f"Starting in 5 seconds…"
            )
            self._wait_precise(5.0)

            # ---- Brief explanation of imagination phase ----
            stimulus.show_instruction(
                "IMAGINATION trials:\n"
                "When prompted, close your eyes and vividly\n"
                "imagine the named color filling your vision.\n\n"
                "IDLE trials: just relax — let your mind wander.\n\n"
                "Continuing in 6 seconds…"
            )
            self._wait_precise(6.0)

            dataset = []

            for trial_idx, trial in enumerate(trials):
                trial_num  = trial_idx + 1
                color_name = trial['color']
                label      = trial['label']
                trial_type = trial['trial_type']

                print(f"\n--- Trial {trial_num}/{total}: {color_name.upper()} [{trial_type}] ---")

                self.sync_manager.reset()
                stimulus.show_progress(trial_num, total, f"{color_name} [{trial_type}]")

                # --- Preparation / fixation ---
                stimulus.show_fixation_cross()
                stimulus.show_instruction("Get Ready…")
                self._wait_precise(1.0)

                for i in range(int(Config.PREPARATION_TIME), 0, -1):
                    stimulus.show_countdown(i)
                    self._wait_precise(1.0)
                stimulus.hide_countdown()

                # --- Flush buffer before stimulus ---
                self.board.get_board_data()
                self._wait_precise(0.1)

                # --- Show stimulus (type-dependent) ---
                if trial_type == 'visual':
                    stimulus_start = stimulus.show_visual_stimulus(color_name)
                elif trial_type == 'imagination':
                    stimulus_start = stimulus.show_imagination_stimulus(color_name)
                else:  # idle
                    stimulus_start = stimulus.show_idle_stimulus()

                stimulus.play_beep()

                try:
                    self.board.insert_marker(label + 1)
                except Exception:
                    pass

                print(f"  Onset: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
                      f"  |  Type: {trial_type}")

                # --- Collect EEG data ---
                trial_data = self._collect_trial_data_sync(Config.TRIAL_DURATION)

                # --- Process & store ---
                if trial_data.shape[1] > 0:
                    raw_path = raw_saver.save_trial(
                        trial_data, color_name, label, trial_type, good_channels,
                        {'trial_number': trial_num,
                         'timestamp': datetime.now().isoformat(),
                         'stimulus_start': stimulus_start}
                    )
                    print(f"  Raw saved: {os.path.basename(raw_path)}")

                    processed = self.feature_extractor.preprocess_data(
                        trial_data, good_channels)
                    features  = self.feature_extractor.extract_comprehensive_features(
                        processed, good_channels)

                    row = {
                        'subject_id':  self.subject_id,
                        'session_name': session_name,
                        'trial_number': trial_num,
                        'color':        color_name,
                        'label':        label,
                        'trial_type':   trial_type,
                        'timestamp':    datetime.now().isoformat(),
                        'channels_used': str(good_channels),
                        'n_samples':    trial_data.shape[1],
                        'n_features':   len(features),
                    }
                    for i, f in enumerate(features):
                        row[f'feature_{i:03d}'] = f
                    dataset.append(row)
                    stimulus.play_completion()
                else:
                    print(f"  WARNING: No data for trial {trial_num}")

                # --- Rest ---
                stimulus.show_rest_screen()
                self._wait_precise(Config.REST_DURATION)

                # Periodic impedance check
                if trial_num % 30 == 0:
                    print("\nPeriodic impedance check…")
                    stimulus.show_instruction("Checking electrodes…")
                    self._wait_precise(1.0)
                    cur_good = self.impedance_monitor.get_good_channels()
                    if len(cur_good) < len(good_channels):
                        print("WARNING: Impedance may have degraded!")
                        stimulus.show_instruction(
                            "Electrode quality may have dropped.\n"
                            "Adjust if needed — continuing in 5 s…")
                        self._wait_precise(5.0)

            # ---- Wrap up ----
            self.board.stop_stream()
            self.impedance_monitor.save_impedance_log()

            if dataset:
                df = pd.DataFrame(dataset)
                fname = f"4colors_{self.subject_id}_{session_name}.csv"
                fpath = os.path.join(Config.DATA_DIR, fname)
                df.to_csv(fpath, index=False)
                df.to_csv(os.path.join(Config.PROCESSED_DATA_DIR, fname), index=False)

                print(f"\n{'='*60}")
                print(f"  Session complete!  {len(df)} trials saved → {fpath}")
                print(f"{'='*60}")
                for cls in Config.COLOR_NAMES:
                    n = len(df[df['color'] == cls])
                    print(f"  {cls.upper():<8}: {n} trials  "
                          f"(visual: {len(df[(df['color']==cls)&(df['trial_type']=='visual')])}, "
                          f"imagine: {len(df[(df['color']==cls)&(df['trial_type']=='imagination')])})")

                stimulus.show_instruction(
                    f"Session Complete!\n\n"
                    f"{len(df)} trials collected\n\n"
                    f"Data saved successfully")
                self._wait_precise(5.0)
                return df
            else:
                print("ERROR: No data collected!")
                stimulus.show_instruction("No data collected!\nCheck your setup.")
                self._wait_precise(5.0)
                return None

        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()
            stimulus.show_instruction(f"Error: {e}\n\nSession aborted")
            self._wait_precise(5.0)
            return None
        finally:
            try:
                self.board.stop_stream()
            except Exception:
                pass
            stimulus.close()


# ==============================
# MODEL TRAINING MANAGER
# ML params matched to red_blue_classifier
# ==============================
class ModelTrainingManager:
    """
    Trains four classifiers (same hyperparameters as red_blue_classifier)
    for the 5-class problem: red · blue · green · yellow · idle.

    Optionally trains separate models for visual-only and imagination-only
    trials to allow mode-specific inference.
    """

    def __init__(self):
        # ---- Matched to red_blue_classifier ----
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, max_depth=10,
                random_state=Config.RANDOM_STATE, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=6,
                random_state=Config.RANDOM_STATE
            ),
            'svm': SVC(
                kernel='rbf', C=1.0, gamma='scale',
                random_state=Config.RANDOM_STATE, probability=True
            ),
            'logistic_regression': LogisticRegression(
                C=1.0, random_state=Config.RANDOM_STATE,
                max_iter=1000, multi_class='multinomial'
            )
        }

    def prepare_data(self, dataframes: List[pd.DataFrame],
                     trial_type_filter: Optional[str] = None
                     ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Combine DataFrames, optionally filter by trial_type, and return X, y.
        trial_type_filter: 'visual' | 'imagination' | 'idle' | None (all)
        """
        combined = pd.concat(dataframes, ignore_index=True)

        if trial_type_filter:
            if trial_type_filter == 'idle':
                combined = combined[combined['trial_type'].isin(['idle', 'visual'])]
            else:
                combined = combined[combined['trial_type'] == trial_type_filter]

        feature_cols = [c for c in combined.columns if c.startswith('feature_')]
        if not feature_cols:
            raise ValueError("No feature columns found in data!")

        X = combined[feature_cols].values
        y = combined['label'].values.astype(int)

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y, combined = X[mask], y[mask], combined[mask].reset_index(drop=True)

        print(f"\n{'='*60}")
        print(f"  Data Preparation  (filter={trial_type_filter or 'all'})")
        print(f"{'='*60}")
        print(f"  Total samples : {X.shape[0]}")
        print(f"  Features      : {X.shape[1]}")
        print(f"\n  Class distribution:")
        for cls, info in Config.COLORS.items():
            n = int(np.sum(y == info['label']))
            print(f"    {cls.upper():<8}: {n} samples")

        return X, y, combined

    def train_and_evaluate_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train and cross-validate all models (matched methodology to red_blue)."""
        if len(X) < 20:
            raise ValueError("Need at least 20 samples to train.")

        results = {}
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=min(Config.TEST_SIZE, 0.3),
            random_state=Config.RANDOM_STATE,
            stratify=y
        )

        print(f"\n  Training : {X_train.shape[0]} samples")
        print(f"  Test     : {X_test.shape[0]} samples\n")

        for name, base_model in self.models.items():
            print(f"--- {name} ---")
            try:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', base_model)
                ])

                # Cross-validation (matched to red_blue: min(CV_FOLDS, n//2, 3))
                cv_folds = min(Config.CV_FOLDS, len(X_train) // 2, 3)
                if cv_folds >= 2:
                    cv_scores = cross_val_score(
                        pipeline, X_train, y_train,
                        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                           random_state=Config.RANDOM_STATE),
                        scoring='accuracy'
                    )
                    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
                else:
                    cv_mean = cv_std = 0.0
                    print("  Not enough data for CV.")

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                test_acc = accuracy_score(y_test, y_pred)

                report = classification_report(
                    y_test, y_pred,
                    target_names=Config.COLOR_NAMES,
                    zero_division=0,
                    output_dict=True
                )

                results[name] = {
                    'model':                  pipeline,
                    'cv_mean':                cv_mean,
                    'cv_std':                 cv_std,
                    'test_accuracy':          test_acc,
                    'confusion_matrix':       confusion_matrix(y_test, y_pred),
                    'classification_report':  report,
                }
                print(f"  CV  : {cv_mean:.3f} ± {cv_std*2:.3f}")
                print(f"  Test: {test_acc:.3f}\n")

            except Exception as e:
                print(f"  Error training {name}: {e}")

        return results

    def print_detailed_results(self, results: Dict):
        print(f"\n{'='*60}")
        print("  MODEL COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Model':<25} {'CV Acc':>10} {'Test Acc':>10}")
        print("  " + "-"*45)
        for name, r in sorted(results.items(), key=lambda x: -x[1]['test_accuracy']):
            print(f"  {name:<25} {r['cv_mean']:>10.3f} {r['test_accuracy']:>10.3f}")

        best = max(results, key=lambda x: results[x]['test_accuracy'])
        br   = results[best]
        print(f"\n  BEST: {best}  (Test acc = {br['test_accuracy']:.3f})")

        print("\n  Confusion Matrix:")
        header = f"  {'':>10}" + "".join(f"{c[:5]:>8}" for c in Config.COLOR_NAMES)
        print(header)
        cm = br['confusion_matrix']
        for i, cls in enumerate(Config.COLOR_NAMES):
            row = f"  {cls:<10}" + "".join(
                f"{cm[i,j]:>8}" if i < cm.shape[0] and j < cm.shape[1] else f"{'N/A':>8}"
                for j in range(len(Config.COLOR_NAMES))
            )
            print(row)

        print("\n  Per-class:")
        for cls in Config.COLOR_NAMES:
            if cls in br['classification_report']:
                m = br['classification_report'][cls]
                print(f"    {cls.upper():<8}  "
                      f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1-score']:.3f}")

    def save_best_model(self, results: Dict, subject_id: str = "general",
                        tag: str = "") -> str:
        if not results:
            raise ValueError("No models trained.")

        best_name  = max(results, key=lambda x: results[x]['test_accuracy'])
        best_model = results[best_name]['model']

        model_info = {
            'model':                 best_model,
            'model_type':            best_name,
            'cv_accuracy':           results[best_name]['cv_mean'],
            'test_accuracy':         results[best_name]['test_accuracy'],
            'confusion_matrix':      results[best_name]['confusion_matrix'],
            'classification_report': results[best_name]['classification_report'],
            'subject_id':            subject_id,
            'training_date':         datetime.now().isoformat(),
            'n_classes':             len(Config.COLORS),
            'class_names':           Config.COLOR_NAMES,
            'config':                {k: v for k, v in Config.__dict__.items()
                                      if not k.startswith('_')},
        }

        setup_directories()
        suffix = f"_{tag}" if tag else ""
        fname  = f"4colors_{subject_id}{suffix}_model.pkl"
        fpath  = os.path.join(Config.MODEL_DIR, fname)
        joblib.dump(model_info, fpath)

        print(f"\n  Model saved → {fpath}")
        print(f"  Type: {best_name}")
        print(f"  Test accuracy: {results[best_name]['test_accuracy']:.3f}")
        return fpath


# ==============================
# REAL-TIME PREDICTION SYSTEM
# Matched vote-based smoothing from red_blue_classifier
# ==============================
class RealTimePredictionSystem:
    """
    Loads a trained model and performs real-time 5-class prediction.
    Uses the same vote-window smoothing as red_blue_classifier.
    """

    def __init__(self, model_path: str, board: BoardShim):
        self.board      = board
        self.model_info = joblib.load(model_path)
        self.model      = self.model_info['model']
        self.extractor  = AdvancedFeatureExtractor()
        self.channels   = BoardShim.get_eeg_channels(board.get_board_id())
        self.class_names = self.model_info.get('class_names', Config.COLOR_NAMES)

        # Vote-based smoothing window (matched to red_blue_classifier)
        self._vote_window: deque = deque(maxlen=Config.PREDICTION_SMOOTHING)

        print(f"\n{'='*60}")
        print(f"  4-Color + Idle Model Loaded")
        print(f"{'='*60}")
        print(f"  Type     : {self.model_info['model_type']}")
        print(f"  Test acc : {self.model_info['test_accuracy']:.3f}")
        print(f"  Classes  : {self.class_names}")

    # ------------------------------------------------------------------
    def predict_from_current_data(self) -> Tuple[str, float, Dict[str, float]]:
        """Return (raw_class, confidence, per_class_probs)."""
        win = int(Config.WINDOW_SIZE * Config.SAMPLING_RATE)
        try:
            data = self.board.get_current_board_data(win)
        except Exception:
            return "no_data", 0.0, {}
        if data.shape[1] < win // 2:
            return "insufficient_data", 0.0, {}

        recent = data[:, -min(win, data.shape[1]):]
        try:
            processed = self.extractor.preprocess_data(recent, self.channels)
            features  = self.extractor.extract_comprehensive_features(processed, self.channels)
            pred_idx  = self.model.predict(features.reshape(1, -1))[0]
            try:
                probs      = self.model.predict_proba(features.reshape(1, -1))[0]
                class_probs = {self.class_names[i]: float(p) for i, p in enumerate(probs)}
                confidence  = float(np.max(probs))
            except Exception:
                class_probs = {}
                confidence  = 0.5
            return self.class_names[pred_idx], confidence, class_probs
        except Exception as e:
            print(f"Prediction error: {e}")
            return "error", 0.0, {}

    def smooth_prediction(self, raw_pred: str) -> str:
        """
        Vote-based smoothing: return the majority class over the last
        PREDICTION_SMOOTHING frames (matched to red_blue_classifier).
        """
        self._vote_window.append(raw_pred)
        valid = [p for p in self._vote_window
                 if p not in ("no_data", "insufficient_data", "error")]
        if not valid:
            return raw_pred
        return max(set(valid), key=valid.count)

    # ------------------------------------------------------------------
    def run_real_time_demo(self):
        """Real-time prediction with colour feedback on screen."""
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        stimulus = FourColorStimulus()

        self.board.start_stream()
        time.sleep(2)

        print("Real-time prediction started. Press ESC to stop.")
        stimulus.show_instruction("Real-time Prediction\nPress ESC to stop")
        time.sleep(2)

        try:
            while True:
                raw, confidence, class_probs = self.predict_from_current_data()
                smoothed = self.smooth_prediction(raw)

                if raw not in ("insufficient_data", "no_data", "error"):
                    if smoothed == 'idle':
                        stimulus.show_idle_stimulus()
                    else:
                        stimulus.show_visual_stimulus(smoothed)

                    prob_str = " | ".join(
                        f"{c[:1].upper()}:{p:.0%}"
                        for c, p in sorted(class_probs.items(), key=lambda x: -x[1])
                    )
                    print(f"  Smoothed: {smoothed:<8}  raw: {raw:<8}  "
                          f"conf: {confidence:.2f}  [{prob_str}]")
                else:
                    stimulus._set_background((60, 60, 60))
                    msg = {"insufficient_data": "Collecting data…",
                           "no_data": "No signal…"}.get(raw, "")
                    if msg:
                        stimulus.show_instruction(msg)

                time.sleep(0.3)
                app.processEvents()

        except KeyboardInterrupt:
            print("\nStopping…")
        finally:
            self.board.stop_stream()
            stimulus.close()


# ==============================
# MAIN FUNCTIONS
# ==============================
def collect_data_session():
    parser = argparse.ArgumentParser(
        description="Collect EEG data — 4-color + idle classifier")
    parser.add_argument('--subject-id',   type=str, required=True)
    parser.add_argument('--session-name', type=str, default=None)
    args = parser.parse_args()

    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = Config.SERIAL_PORT
    board = BoardShim(Config.BOARD_ID, params)
    try:
        board.prepare_session()
        collector = DataCollectionManager(board, args.subject_id)
        df = collector.collect_session_data(args.session_name)
        if df is not None:
            print(f"\nDone!  {len(df)} trials.")
    except Exception as e:
        import traceback; traceback.print_exc()
    finally:
        if board.is_prepared():
            board.release_session()


def train_models():
    """
    Train 5-class models.
    By default trains a combined model (visual + imagination).
    With --mode visual|imagination|all, trains on that subset only.
    """
    parser = argparse.ArgumentParser(
        description="Train 4-color + idle classification models")
    parser.add_argument('--data-dir',   type=str, default=Config.DATA_DIR)
    parser.add_argument('--subject-id', type=str, default='general')
    parser.add_argument('--mode',       type=str, default='all',
                        choices=['all', 'visual', 'imagination'],
                        help="Filter trials by type before training")
    args = parser.parse_args()

    csv_files = [f for f in os.listdir(args.data_dir)
                 if f.startswith('4colors_') and f.endswith('.csv')]

    if not csv_files:
        print(f"No 4colors_*.csv files found in {args.data_dir}")
        return

    print(f"Found {len(csv_files)} data files")
    dataframes = []
    for f in csv_files:
        try:
            df = pd.read_csv(os.path.join(args.data_dir, f))
            dataframes.append(df)
            print(f"  Loaded {f}: {len(df)} trials")
        except Exception as e:
            print(f"  Error loading {f}: {e}")

    if not dataframes:
        print("No valid data files!"); return

    try:
        trainer = ModelTrainingManager()
        filt = None if args.mode == 'all' else args.mode
        X, y, _ = trainer.prepare_data(dataframes, trial_type_filter=filt)
        results  = trainer.train_and_evaluate_models(X, y)

        if results:
            trainer.print_detailed_results(results)
            trainer.save_best_model(results, args.subject_id,
                                    tag=args.mode if args.mode != 'all' else '')
        else:
            print("No models trained successfully.")

    except Exception as e:
        import traceback; traceback.print_exc()


def run_real_time_prediction():
    parser = argparse.ArgumentParser(
        description="Real-time 4-color + idle prediction")
    parser.add_argument('--model-path', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}"); return

    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = Config.SERIAL_PORT
    board = BoardShim(Config.BOARD_ID, params)
    try:
        board.prepare_session()
        predictor = RealTimePredictionSystem(args.model_path, board)
        predictor.run_real_time_demo()
    except Exception as e:
        import traceback; traceback.print_exc()
    finally:
        if board.is_prepared():
            board.release_session()


# ==============================
# MAIN ENTRY POINT
# ==============================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("""
╔══════════════════════════════════════════════════════════════════╗
║          4-COLOR + IDLE EEG CLASSIFIER                           ║
║     Red · Blue · Green · Yellow · Idle                           ║
║     Visual  ·  Imagination  ·  Idle trials                       ║
╠══════════════════════════════════════════════════════════════════╣
║ USAGE:                                                           ║
║                                                                  ║
║ 1. COLLECT DATA:                                                 ║
║    python3 4colors_classifier.py collect                         ║
║            --subject-id NAME [--session-name SESSION]            ║
║                                                                  ║
║ 2. TRAIN MODEL (all trial types):                                ║
║    python3 4colors_classifier.py train                           ║
║            [--data-dir DIR] [--subject-id ID] [--mode all]       ║
║                                                                  ║
║    Train on visual trials only:                                  ║
║    python3 4colors_classifier.py train --mode visual             ║
║                                                                  ║
║    Train on imagination trials only:                             ║
║    python3 4colors_classifier.py train --mode imagination        ║
║                                                                  ║
║ 3. REAL-TIME PREDICTION:                                         ║
║    python3 4colors_classifier.py predict                         ║
║            --model-path 4colors_models/4colors_alice_model.pkl   ║
╠══════════════════════════════════════════════════════════════════╣
║ CLASSES:  red(0) · blue(1) · green(2) · yellow(3) · idle(4)     ║
║ TRIAL TYPES:  visual · imagination · idle                        ║
╚══════════════════════════════════════════════════════════════════╝
        """)
        sys.exit(1)

    command  = sys.argv[1]
    sys.argv = sys.argv[1:]

    if command == "collect":
        collect_data_session()
    elif command == "train":
        train_models()
    elif command == "predict":
        run_real_time_prediction()
    else:
        print(f"Unknown command: {command}  (valid: collect · train · predict)")
        sys.exit(1)


# ==============================
# QUICK REFERENCE
# ==============================
# Collect:
#   python3 4colors_classifier.py collect --subject-id nam --session-name s2
#   python3 4colors_classifier.py collect --subject-id luan --session-name s1
#
# Train (combined):
#   python3 4colors_classifier.py train --subject-id nam --mode all
# Predict:
#   python3 4colors_classifier.py predict --model-path 4colors_models/4colors_nam_model.pkl