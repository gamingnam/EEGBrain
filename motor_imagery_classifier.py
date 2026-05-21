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
    TRIAL_DURATION = 4.0       # seconds per cue
    REST_DURATION = 2.0        # seconds between trials
    PREPARATION_TIME = 2.0     # fixation cross before cue
    TRIALS_PER_CLASS = 2       # trials per class label (increase for better training)

    # Data Processing
    SAMPLING_RATE = 250        # Cyton board sampling rate
    #NUM_CHANNELS = 8           # Cyton board EEG channels
    FILTER_LOW = 0.5           # Hz — high-pass filter
    FILTER_HIGH = 40.0         # Hz — low-pass filter
    NOTCH_FREQ = 60.0          # Hz — US powerline; change to 50 for EU

    # Feature Extraction
    WINDOW_SIZE = 2.0          # seconds for feature extraction
    OVERLAP = 0.5              # window overlap

    # Classes
    TASK_MAP = {
        "task1": ["left_fist",  "right_fist"],
        "task2": ["left_fist",  "right_fist"],
        "task3": ["both_fists", "both_feet"],
        "task4": ["both_fists", "both_feet"],
    }
    ALL_CLASSES = ["left_fist", "right_fist", "both_fists", "both_feet"]
    NUM_CLASSES = len(ALL_CLASSES)

    # Directories
    DATA_DIR = "motor_imagery_data"
    MODEL_DIR = "models"

    # Training
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    RANDOM_STATE = 42

    # Full experimental protocol run order
    PROTOCOL_RUNS = [
        ("baseline_eyes_open",   None),
        ("baseline_eyes_closed", None),
        ("task1", False),
        ("task2", True),
        ("task3", False),
        ("task4", True),
    ]


# ==============================
# AUDIO FEEDBACK
# ==============================
class AudioFeedback:
    @staticmethod
    def beep():
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


# ==============================
# STIMULUS WINDOW
# ==============================
class MotorImageryStimulus(QtWidgets.QWidget):
    """
    Fullscreen stimulus window displaying fixation cross, cue arrows,
    rest screen, and instruction overlays.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motor Imagery EEG")
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint)
        self.showFullScreen()
        self._cue = None
        self._phase = "blank"
        self.audio = AudioFeedback()

    # ── painting ──────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        w, h = self.width(), self.height()

        bg = {"fixation": "#1a1a2e", "cue": "#0f3460",
              "rest": "#16213e", "blank": "#000000"}.get(self._phase, "#000000")
        painter.fillRect(self.rect(), QtGui.QColor(bg))

        if self._phase == "fixation":
            self._draw_fixation(painter, w, h)
        elif self._phase == "cue" and self._cue:
            self._draw_fixation(painter, w, h)
            self._draw_target_arrow(painter, w, h, self._cue)
        elif self._phase == "rest":
            self._draw_text(painter, w, h, "RELAX", "#4ecdc4", 52)

        painter.end()

    def _draw_fixation(self, painter, w, h):
        pen = QtGui.QPen(QtGui.QColor("#e0e0e0"), 4)
        painter.setPen(pen)
        cx, cy, arm = w // 2, h // 2, 30
        painter.drawLine(cx - arm, cy, cx + arm, cy)
        painter.drawLine(cx, cy - arm, cx, cy + arm)

    def _draw_target_arrow(self, painter, w, h, direction):
        color = QtGui.QColor("#f5a623")
        pen = QtGui.QPen(color, 6)
        painter.setPen(pen)
        painter.setBrush(color)

        cx, cy = w // 2, h // 2
        sz = 60
        margin = 80

        dirs = {
            "left":   (margin + sz,       cy,           -1,  0),
            "right":  (w - margin - sz,   cy,            1,  0),
            "top":    (cx,                margin + sz,   0, -1),
            "bottom": (cx,                h - margin - sz, 0,  1),
        }
        if direction not in dirs:
            return

        tx, ty, dx, dy = dirs[direction]
        pts = QtGui.QPolygon([
            QtCore.QPoint(int(tx + dx * sz),        int(ty + dy * sz)),
            QtCore.QPoint(int(tx - dy * sz // 2),   int(ty + dx * sz // 2)),
            QtCore.QPoint(int(tx + dy * sz // 2),   int(ty - dx * sz // 2)),
        ])
        painter.drawPolygon(pts)

        font = QtGui.QFont("Courier New", 22, QtGui.QFont.Bold)
        painter.setFont(font)
        label_map = {
            "left":   "← LEFT FIST",
            "right":  "RIGHT FIST →",
            "top":    "↑ BOTH FISTS",
            "bottom": "↓ BOTH FEET",
        }
        text = label_map.get(direction, direction.upper())
        painter.drawText(self.rect(), QtCore.Qt.AlignCenter | QtCore.Qt.AlignBottom
                         if dy < 0 else QtCore.Qt.AlignCenter, text)

    def _draw_text(self, painter, w, h, text, hex_color, font_size):
        font = QtGui.QFont("Courier New", font_size, QtGui.QFont.Bold)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(hex_color))
        painter.drawText(self.rect(), QtCore.Qt.AlignCenter, text)

    # ── public API ────────────────────────────────────────────────────────────
    def show_blank(self):
        self._phase = "blank"
        self._cue = None
        self.update()
        QtWidgets.QApplication.processEvents()

    def show_fixation(self):
        self._phase = "fixation"
        self._cue = None
        self.update()
        QtWidgets.QApplication.processEvents()
        self.audio.beep()

    def show_cue(self, direction: str):
        self._phase = "cue"
        self._cue = direction
        self.update()
        QtWidgets.QApplication.processEvents()
        self.audio.beep()

    def show_rest(self):
        self._phase = "rest"
        self._cue = None
        self.update()
        QtWidgets.QApplication.processEvents()
        self.audio.beep()

    def show_message(self, msg: str):
        self._phase = "blank"
        self._cue = None
        self.update()
        label = QtWidgets.QLabel(msg, self)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setStyleSheet("""
            QLabel {
                font: bold 26px 'Courier New';
                color: #e0e0e0;
                background: rgba(0,0,0,160);
                padding: 20px;
                border-radius: 10px;
            }
        """)
        label.resize(600, 200)
        label.move(self.width() // 2 - 300, self.height() // 2 - 100)
        label.show()
        QtWidgets.QApplication.processEvents()
        time.sleep(3)
        label.deleteLater()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()


# ==============================
# FEATURE EXTRACTION
# ==============================
class FeatureExtractor:
    """
    Extracts the same comprehensive time + frequency domain features
    used in red_blue_classifier, applied per EEG channel.
    """

    def __init__(self, sampling_rate: int = Config.SAMPLING_RATE):
        self.sampling_rate = sampling_rate

    def preprocess_channel(self, channel_data: np.ndarray) -> np.ndarray:
        """Band-pass + notch filter a single channel (in-place, returns copy)."""
        sample = np.copy(channel_data, order='C')
        try:
            DataFilter.detrend(sample, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(
                sample, self.sampling_rate,
                Config.FILTER_LOW, Config.FILTER_HIGH, 4,
                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
            )
            DataFilter.perform_bandstop(
                sample, self.sampling_rate,
                Config.NOTCH_FREQ - 2, Config.NOTCH_FREQ + 2, 4,
                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
            )
            DataFilter.remove_environmental_noise(
                sample, self.sampling_rate, NoiseTypes.SIXTY.value
            )
        except Exception as e:
            print(f"  [warn] preprocess_channel: {e}")
        return sample

    def extract_features_from_raw(self,
                                   raw_data: np.ndarray,
                                   channel_indices: List[int]) -> np.ndarray:
        """
        Given raw board data (shape: board_rows × samples), preprocess
        each EEG channel and extract 18 features per channel.
        Returns a flat feature vector.
        """
        features = []
        for ch in channel_indices:
            if ch >= raw_data.shape[0] or raw_data.shape[1] == 0:
                features.extend([0.0] * 18)
                continue

            channel_data = self.preprocess_channel(raw_data[ch])

            try:
                # ── time domain (6 features) ──────────────────────────────
                features.extend([
                    float(np.mean(channel_data)),
                    float(np.std(channel_data)),
                    float(np.max(channel_data) - np.min(channel_data)),  # peak-to-peak
                    float(np.var(channel_data)),
                    float(np.median(channel_data)),
                    float(np.percentile(channel_data, 75) - np.percentile(channel_data, 25)),  # IQR
                ])

                # ── frequency domain (12 features) ───────────────────────
                try:
                    psd = DataFilter.get_psd_welch(
                        channel_data,
                        DataFilter.get_nearest_power_of_two(self.sampling_rate),
                        self.sampling_rate // 2,
                        self.sampling_rate,
                        WindowOperations.BLACKMAN_HARRIS.value
                    )

                    delta = DataFilter.get_band_power(psd, 0.5,  4.0)
                    theta = DataFilter.get_band_power(psd, 4.0,  8.0)
                    alpha = DataFilter.get_band_power(psd, 8.0,  12.0)
                    beta  = DataFilter.get_band_power(psd, 12.0, 30.0)
                    gamma = DataFilter.get_band_power(psd, 30.0, 40.0)
                    total = delta + theta + alpha + beta + gamma

                    # relative band powers
                    features.extend([
                        delta / total if total > 0 else 0.0,
                        theta / total if total > 0 else 0.0,
                        alpha / total if total > 0 else 0.0,
                        beta  / total if total > 0 else 0.0,
                        gamma / total if total > 0 else 0.0,
                    ])

                    # band ratios
                    features.extend([
                        alpha / beta  if beta  > 0 else 0.0,
                        theta / alpha if alpha > 0 else 0.0,
                        beta  / gamma if gamma > 0 else 0.0,
                    ])

                    # motor-imagery-relevant bands
                    mu_band    = DataFilter.get_band_power(psd, 8.0,  13.0)  # mu rhythm
                    beta_motor = DataFilter.get_band_power(psd, 13.0, 30.0)  # motor beta

                    features.extend([
                        mu_band    / total if total > 0 else 0.0,
                        beta_motor / total if total > 0 else 0.0,
                    ])

                except Exception as e:
                    print(f"  [warn] PSD failed for ch {ch}: {e}")
                    features.extend([0.0] * 12)

            except Exception as e:
                print(f"  [warn] Feature extraction failed for ch {ch}: {e}")
                features.extend([0.0] * 18)

        return np.array(features, dtype=np.float32)

    def extract_features_from_segment_row(self,
                                           row: pd.Series,
                                           num_channels: int,
                                           window_samples: int) -> np.ndarray:
        """
        Re-extract features from a previously saved segment stored as
        flat columns t000_ch0 … tNNN_chM in a CSV row.
        Returns a flat feature vector identical in structure to
        extract_features_from_raw().
        """
        features = []
        for c in range(num_channels):
            col_vals = []
            for t in range(window_samples):
                col = f"t{t:03d}_ch{c}"
                col_vals.append(float(row.get(col, 0.0)))
            channel_data = np.array(col_vals, dtype=np.float64)

            try:
                # time domain
                features.extend([
                    float(np.mean(channel_data)),
                    float(np.std(channel_data)),
                    float(np.max(channel_data) - np.min(channel_data)),
                    float(np.var(channel_data)),
                    float(np.median(channel_data)),
                    float(np.percentile(channel_data, 75) - np.percentile(channel_data, 25)),
                ])

                # frequency domain
                try:
                    psd = DataFilter.get_psd_welch(
                        channel_data,
                        DataFilter.get_nearest_power_of_two(self.sampling_rate),
                        self.sampling_rate // 2,
                        self.sampling_rate,
                        WindowOperations.BLACKMAN_HARRIS.value
                    )
                    delta = DataFilter.get_band_power(psd, 0.5,  4.0)
                    theta = DataFilter.get_band_power(psd, 4.0,  8.0)
                    alpha = DataFilter.get_band_power(psd, 8.0,  12.0)
                    beta  = DataFilter.get_band_power(psd, 12.0, 30.0)
                    gamma = DataFilter.get_band_power(psd, 30.0, 40.0)
                    total = delta + theta + alpha + beta + gamma

                    features.extend([
                        delta / total if total > 0 else 0.0,
                        theta / total if total > 0 else 0.0,
                        alpha / total if total > 0 else 0.0,
                        beta  / total if total > 0 else 0.0,
                        gamma / total if total > 0 else 0.0,
                        alpha / beta  if beta  > 0 else 0.0,
                        theta / alpha if alpha > 0 else 0.0,
                        beta  / gamma if gamma > 0 else 0.0,
                        DataFilter.get_band_power(psd, 8.0,  13.0) / total if total > 0 else 0.0,
                        DataFilter.get_band_power(psd, 13.0, 30.0) / total if total > 0 else 0.0,
                    ])
                except Exception:
                    features.extend([0.0] * 12)

            except Exception:
                features.extend([0.0] * 18)

        return np.array(features, dtype=np.float32)


# ==============================
# DATA COLLECTION MANAGER
# ==============================
class DataCollectionManager:
    def __init__(self, board: BoardShim, subject_id: str):
        self.board = board
        self.subject_id = subject_id
        self.eeg_channels = BoardShim.get_eeg_channels(Config.BOARD_ID)[:Config.NUM_CHANNELS]
        self.feature_extractor = FeatureExtractor()
        os.makedirs(Config.DATA_DIR, exist_ok=True)

    # ── direction helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _class_to_direction(class_name: str) -> str:
        return {
            "left_fist":  "left",
            "right_fist": "right",
            "both_fists": "top",
            "both_feet":  "bottom",
        }[class_name]

    # ── baseline collection ───────────────────────────────────────────────────
    def collect_baseline(self, stimulus: MotorImageryStimulus,
                         eyes_open: bool, duration: float = 30.0) -> pd.DataFrame:
        label_name = "baseline_eyes_open" if eyes_open else "baseline_eyes_closed"
        instruction = ("Keep eyes OPEN and relax.\nDo not move." if eyes_open
                       else "Close your eyes and relax.\nDo not move.")

        print(f"\n=== Baseline: {label_name} ({duration}s) ===")
        stimulus.show_message(instruction)
        stimulus.show_fixation() if eyes_open else stimulus.show_rest()

        self.board.start_stream()
        time.sleep(duration)
        data = self.board.get_board_data()
        self.board.stop_stream()

        # Slide windows and extract features (baselines excluded from training by default)
        records = []
        step = Config.SAMPLING_RATE  # 1-second steps
        window_samples = int(Config.WINDOW_SIZE * Config.SAMPLING_RATE)
        num_windows = max(0, (data.shape[1] - window_samples) // step + 1)

        for w in range(num_windows):
            start = w * step
            chunk = data[:, start:start + window_samples]
            features = self.feature_extractor.extract_features_from_raw(
                chunk, self.eeg_channels
            )
            rec = {
                "subject_id":  self.subject_id,
                "label":       label_name,
                "label_idx":   -1,
                "window":      w,
                "timestamp":   datetime.now().isoformat(),
            }
            for i, v in enumerate(features):
                rec[f"feature_{i:03d}"] = v
            records.append(rec)

        df = pd.DataFrame(records)
        self._save_df(df, label_name)
        print(f"  Saved {len(df)} baseline windows.")
        return df

    # ── task collection ───────────────────────────────────────────────────────
    def collect_task_run(self, stimulus: MotorImageryStimulus,
                         task_name: str, imagined: bool) -> pd.DataFrame:
        classes = Config.TASK_MAP[task_name]
        mode = "IMAGINE" if imagined else "PERFORM"
        print(f"\n=== {task_name.upper()} ({mode}) — classes: {classes} ===")
        stimulus.show_message(
            f"{'IMAGINE' if imagined else 'PERFORM'} the movement\n"
            f"shown by the arrow.\n\n{task_name.upper()}"
        )

        trials = classes * Config.TRIALS_PER_CLASS
        random.shuffle(trials)

        records = []
        self.board.start_stream()
        time.sleep(1)

        for idx, class_name in enumerate(trials):
            print(f"  Trial {idx + 1}/{len(trials)}: {class_name} [{mode}]")
            direction = self._class_to_direction(class_name)
            label_idx = Config.ALL_CLASSES.index(class_name)

            # preparation
            stimulus.show_fixation()
            time.sleep(Config.PREPARATION_TIME)

            # cue on
            stimulus.show_cue(direction)
            try:
                self.board.insert_marker(float(label_idx + 1))
            except Exception:
                pass

            time.sleep(Config.TRIAL_DURATION)

            # collect window
            num_samples = int(Config.TRIAL_DURATION * Config.SAMPLING_RATE)
            try:
                trial_data = self.board.get_current_board_data(num_samples)
            except Exception:
                trial_data = self.board.get_board_data()

            if trial_data.shape[1] > 0:
                # Extract features directly from raw data
                features = self.feature_extractor.extract_features_from_raw(
                    trial_data, self.eeg_channels
                )
                rec = {
                    "subject_id": self.subject_id,
                    "task":       task_name,
                    "imagined":   imagined,
                    "label":      class_name,
                    "label_idx":  label_idx,
                    "trial":      idx + 1,
                    "timestamp":  datetime.now().isoformat(),
                    "n_features": len(features),
                }
                for i, v in enumerate(features):
                    rec[f"feature_{i:03d}"] = v
                records.append(rec)
            else:
                print(f"  Warning: no data for trial {idx + 1}")

            # rest
            stimulus.show_rest()
            time.sleep(Config.REST_DURATION)

        self.board.stop_stream()
        df = pd.DataFrame(records)
        mode_tag = "imagined" if imagined else "actual"
        self._save_df(df, f"{task_name}_{mode_tag}")
        print(f"  Saved {len(df)} trials.")
        return df

    # ── full protocol ─────────────────────────────────────────────────────────
    def run_full_protocol(self) -> List[pd.DataFrame]:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        stimulus = MotorImageryStimulus()
        all_dfs = []

        try:
            stimulus.show_message(
                "Motor Imagery EEG Protocol\n\n"
                "Follow the on-screen cues.\n"
                "Press ESC at any time to abort.\n\n"
                "Starting in 5 seconds..."
            )
            time.sleep(5)

            for run_idx, (run_type, imagined) in enumerate(Config.PROTOCOL_RUNS):
                print(f"\n{'='*60}")
                print(f"Run {run_idx + 1}/{len(Config.PROTOCOL_RUNS)}: {run_type}")
                print(f"{'='*60}")

                if run_type == "baseline_eyes_open":
                    df = self.collect_baseline(stimulus, eyes_open=True, duration=60.0)
                elif run_type == "baseline_eyes_closed":
                    df = self.collect_baseline(stimulus, eyes_open=False, duration=60.0)
                else:
                    df = self.collect_task_run(stimulus, run_type, imagined)

                if df is not None and len(df) > 0:
                    all_dfs.append(df)

                if run_idx < len(Config.PROTOCOL_RUNS) - 1:
                    stimulus.show_message(
                        f"Run {run_idx + 1} complete.\n\nTake a short break.\n"
                        f"Next run in 10 seconds."
                    )
                    time.sleep(10)

            stimulus.show_message("Protocol Complete!\nThank you.")
        except Exception as e:
            print(f"Error during protocol: {e}")
        finally:
            stimulus.close()

        return all_dfs

    # ── utility ───────────────────────────────────────────────────────────────
    def _save_df(self, df: pd.DataFrame, tag: str):
        filename = (f"{self.subject_id}_{tag}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        path = os.path.join(Config.DATA_DIR, filename)
        df.to_csv(path, index=False)
        print(f"  Saved → {path}")


# ==============================
# MODEL TRAINING MANAGER
# ==============================
class ModelTrainingManager:
    """
    Sklearn-based multi-class training pipeline (mirrors red_blue_classifier).
    Expects CSVs whose feature columns are named feature_000, feature_001, …
    and whose label column is label_idx (0–3 for the four motor imagery classes).
    """

    def __init__(self):
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
            ),
        }

    def prepare_data(self, data_dir: str,
                     exclude_baselines: bool = True
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Load all CSVs, optionally drop baselines, return (X, y)."""
        csv_files = [f for f in os.listdir(data_dir)
                     if f.endswith('.csv') and not f.startswith('impedance')]

        if not csv_files:
            raise FileNotFoundError(f"No data CSVs found in {data_dir}")

        print(f"Loading {len(csv_files)} file(s) from {data_dir} ...")
        dataframes = []

        for fname in csv_files:
            path = os.path.join(data_dir, fname)
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"  Skipping {fname}: {e}")
                continue

            if exclude_baselines and 'label_idx' in df.columns:
                df = df[df['label_idx'].isin(range(Config.NUM_CLASSES))]

            if df.empty:
                continue

            feature_cols = [c for c in df.columns if c.startswith('feature_')]
            if not feature_cols:
                # Legacy format: try to re-extract from raw segment columns
                print(f"  {fname}: no feature_ columns, attempting re-extraction …")
                extractor = FeatureExtractor()
                window_samples = 150  # default from old collector
                rows = []
                for _, row in df.iterrows():
                    if row.get('label_idx', -1) not in range(Config.NUM_CLASSES):
                        continue
                    feats = extractor.extract_features_from_segment_row(
                        row, Config.NUM_CHANNELS, window_samples
                    )
                    r = {"label_idx": int(row['label_idx'])}
                    for i, v in enumerate(feats):
                        r[f"feature_{i:03d}"] = v
                    rows.append(r)
                if rows:
                    df = pd.DataFrame(rows)
                    feature_cols = [c for c in df.columns if c.startswith('feature_')]
                else:
                    print(f"  Skipping {fname}: re-extraction yielded no rows.")
                    continue

            dataframes.append(df[feature_cols + ['label_idx']])
            print(f"  {fname}: {len(df)} rows")

        if not dataframes:
            raise ValueError("No usable data after loading.")

        combined = pd.concat(dataframes, ignore_index=True)
        feature_cols = [c for c in combined.columns if c.startswith('feature_')]

        X = combined[feature_cols].values
        y = combined['label_idx'].values.astype(int)

        # drop NaN / inf
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]

        print(f"\nDataset: {X.shape[0]} samples | {X.shape[1]} features")
        for idx, name in enumerate(Config.ALL_CLASSES):
            print(f"  Class {idx} ({name}): {np.sum(y == idx)} samples")

        return X, y

    def train_and_evaluate(self, X: np.ndarray,
                           y: np.ndarray) -> Dict:
        """Train all models, cross-validate, evaluate on held-out test set."""
        if len(X) < 10:
            raise ValueError("Need at least 10 samples to train.")

        unique_classes = np.unique(y)
        stratify = y if len(unique_classes) > 1 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=min(Config.TEST_SIZE, 0.4),
            random_state=Config.RANDOM_STATE,
            stratify=stratify
        )

        print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")
        results = {}

        for name, base_model in self.models.items():
            print(f"\n--- {name} ---")
            try:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', base_model),
                ])

                cv_folds = min(Config.CV_FOLDS, len(X_train) // max(len(unique_classes), 2))
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
                    print("  Not enough data for cross-validation.")

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                test_acc = accuracy_score(y_test, y_pred)

                results[name] = {
                    'model':                  pipeline,
                    'cv_mean':                cv_mean,
                    'cv_std':                 cv_std,
                    'test_accuracy':          test_acc,
                    'confusion_matrix':       confusion_matrix(y_test, y_pred),
                    'classification_report':  classification_report(
                        y_test, y_pred,
                        target_names=[Config.ALL_CLASSES[i] for i in unique_classes],
                        zero_division=0
                    ),
                }

                print(f"  CV:   {cv_mean:.3f} ± {cv_std * 2:.3f}")
                print(f"  Test: {test_acc:.3f}")
                print(results[name]['classification_report'])

            except Exception as e:
                print(f"  Error training {name}: {e}")

        return results

    def save_best_model(self, results: Dict, subject_id: str = "general") -> str:
        """Save the best model (by test accuracy) as a .pkl file."""
        if not results:
            raise ValueError("No models were successfully trained!")

        best_name = max(results, key=lambda k: results[k]['test_accuracy'])
        best = results[best_name]

        model_info = {
            'model':          best['model'],
            'model_type':     best_name,
            'cv_accuracy':    best['cv_mean'],
            'test_accuracy':  best['test_accuracy'],
            'subject_id':     subject_id,
            'training_date':  datetime.now().isoformat(),
            'classes':        Config.ALL_CLASSES,
            'num_classes':    Config.NUM_CLASSES,
        }

        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        filename = f"{subject_id}_motor_imagery_model.pkl"
        filepath = os.path.join(Config.MODEL_DIR, filename)
        joblib.dump(model_info, filepath)

        print(f"\nBest model ({best_name}) saved → {filepath}")
        print(f"  CV Accuracy:   {best['cv_mean']:.3f}")
        print(f"  Test Accuracy: {best['test_accuracy']:.3f}")
        return filepath


# ==============================
# REAL-TIME PREDICTION SYSTEM
# ==============================
class RealTimePredictionSystem:
    """
    Loads a .pkl sklearn model and runs live classification,
    mirroring red_blue_classifier's predict loop.
    """

    def __init__(self, model_path: str, board: BoardShim):
        self.board = board
        self.model_info = joblib.load(model_path)
        self.model = self.model_info['model']
        self.classes = self.model_info.get('classes', Config.ALL_CLASSES)
        self.eeg_channels = BoardShim.get_eeg_channels(Config.BOARD_ID)[:Config.NUM_CHANNELS]
        self.feature_extractor = FeatureExtractor()

        print(f"Loaded model: {self.model_info['model_type']}")
        print(f"  Training accuracy: {self.model_info['test_accuracy']:.3f}")
        print(f"  Classes: {self.classes}")
        print(f"  EEG channels: {self.eeg_channels}")

    def predict_from_current_data(self) -> Tuple[str, float]:
        """Return (predicted_class_name, confidence)."""
        window_samples = int(Config.WINDOW_SIZE * Config.SAMPLING_RATE)
        try:
            data = self.board.get_current_board_data(
                int(Config.TRIAL_DURATION * Config.SAMPLING_RATE)
            )
        except Exception:
            return "no_data", 0.0

        if data.shape[1] < window_samples // 2:
            return "insufficient_data", 0.0

        try:
            features = self.feature_extractor.extract_features_from_raw(
                data, self.eeg_channels
            )
            pred_idx = self.model.predict(features.reshape(1, -1))[0]
            try:
                confidence = float(self.model.predict_proba(features.reshape(1, -1)).max())
            except Exception:
                confidence = 0.5

            pred_class = (self.classes[pred_idx]
                          if pred_idx < len(self.classes) else "unknown")
            return pred_class, confidence

        except Exception as e:
            print(f"Prediction error: {e}")
            return "error", 0.0

    def run_demo(self):
        """Live prediction with on-screen arrow feedback."""
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        stim = MotorImageryStimulus()

        self.board.start_stream()
        time.sleep(2)

        print("Real-time motor imagery prediction running. Press ESC to stop.")

        direction_map = {
            "left_fist":  "left",
            "right_fist": "right",
            "both_fists": "top",
            "both_feet":  "bottom",
        }

        stim.show_message("Real-time Prediction\nPress ESC to stop")

        # sliding majority-vote buffer
        vote_buffer: List[str] = []
        vote_size = 6

        try:
            while True:
                pred_class, conf = self.predict_from_current_data()

                if pred_class in direction_map:
                    vote_buffer.append(pred_class)
                    if len(vote_buffer) > vote_size:
                        vote_buffer.pop(0)
                    smoothed = max(set(vote_buffer), key=vote_buffer.count)
                    stim.show_cue(direction_map[smoothed])
                    print(f"  {smoothed:12s}  conf={conf:.2f}")
                elif pred_class == "insufficient_data":
                    stim.show_rest()
                    print("  Insufficient data …")
                elif pred_class == "no_data":
                    stim.show_blank()
                    print("  No data …")
                else:
                    stim.show_rest()
                    print(f"  [{pred_class}]")

                time.sleep(0.5)
                app.processEvents()

        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            self.board.stop_stream()
            stim.close()


# ==============================
# CLI ENTRY POINTS
# ==============================
def cmd_collect():
    parser = argparse.ArgumentParser(description="Collect motor imagery EEG data")
    parser.add_argument('--subject-id', required=True, help='e.g. subjectAlice')
    args = parser.parse_args()

    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = Config.SERIAL_PORT
    board = BoardShim(Config.BOARD_ID, params)

    try:
        board.prepare_session()
        collector = DataCollectionManager(board, args.subject_id)
        all_dfs = collector.run_full_protocol()
        print(f"\nProtocol complete. {len(all_dfs)} run(s) saved.")
    finally:
        if board.is_prepared():
            board.release_session()


def cmd_train():
    parser = argparse.ArgumentParser(description="Train sklearn model for motor imagery")
    parser.add_argument('--data-dir', default=Config.DATA_DIR)
    parser.add_argument('--subject-id', default='general')
    parser.add_argument('--include-baselines', action='store_true',
                        help='Include baseline runs in training (not recommended)')
    args = parser.parse_args()

    trainer = ModelTrainingManager()
    X, y = trainer.prepare_data(args.data_dir,
                                 exclude_baselines=not args.include_baselines)
    results = trainer.train_and_evaluate(X, y)

    if results:
        print("\n=== MODEL COMPARISON ===")
        for name, r in results.items():
            print(f"  {name:22s} | CV: {r['cv_mean']:.3f} | Test: {r['test_accuracy']:.3f}")
        trainer.save_best_model(results, args.subject_id)
    else:
        print("No models were successfully trained!")


def cmd_predict():
    parser = argparse.ArgumentParser(description="Real-time motor imagery prediction")
    parser.add_argument('--model-path', required=True, help='Path to .pkl model file')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        sys.exit(1)

    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = Config.SERIAL_PORT
    board = BoardShim(Config.BOARD_ID, params)

    try:
        board.prepare_session()
        predictor = RealTimePredictionSystem(args.model_path, board)
        predictor.run_demo()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if board.is_prepared():
            board.release_session()


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    USAGE = """
Motor Imagery EEG Classifier (sklearn / pkl)
=============================================
Commands:
  collect   — Run the full EEG protocol and save data
  train     — Train sklearn models and save best as .pkl
  predict   — Load a .pkl model and run real-time classification

Examples:
  python3 motor_imagery_classifier.py collect --subject-id subjectAlice
  python3 motor_imagery_classifier.py train --subject-id subjectAlice
  python3 motor_imagery_classifier.py train --subject-id general
  python3 motor_imagery_classifier.py predict --model-path models/subjectAlice_motor_imagery_model.pkl
  python3 motor_imagery_classifier.py predict --model-path models/general_motor_imagery_model.pkl
"""

    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = sys.argv[1:]

    if command == "collect":
        cmd_collect()
    elif command == "train":
        cmd_train()
    elif command == "predict":
        cmd_predict()
    else:
        print(f"Unknown command: {command}\n{USAGE}")
        sys.exit(1)