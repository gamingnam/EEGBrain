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
    TRIAL_DURATION = 4.0  # seconds per color (longer for better features)
    REST_DURATION = 1.5  # seconds between trials
    TRIALS_PER_COLOR = 30  # increased for better training
    PREPARATION_TIME = 2.0  # time before stimulus appears

    # Data Processing
    SAMPLING_RATE = 250  # Cyton board sampling rate
    IMPEDANCE_THRESHOLD = 50000  # microVolts - adjust based on your setup
    FILTER_LOW = 0.5  # Hz - high-pass filter
    FILTER_HIGH = 40.0  # Hz - low-pass filter
    NOTCH_FREQ = 60.0  # Hz - US powerline frequency (50 for EU)

    # Feature Extraction
    WINDOW_SIZE = 2.0  # seconds for feature extraction windows
    OVERLAP = 0.5  # window overlap (50%)

    # Files
    DATA_DIR = "red_blue_data"
    MODEL_DIR = "models"
    IMPEDANCE_LOG = "impedance_log.csv"

    # Training
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    RANDOM_STATE = 42


# ==============================
# AUDIO FEEDBACK UTILITY
# ==============================
class AudioFeedback:
    @staticmethod
    def play_beep():
        """Cross-platform beep function."""
        try:
            # Windows
            if sys.platform == "win32":
                import winsound
                winsound.Beep(1000, 300)  # 1000 Hz, 300 ms
                return
        except ImportError:
            pass

        try:
            # macOS
            if sys.platform == "darwin":
                os.system('afplay /System/Library/Sounds/Pop.aiff 2>/dev/null &')
                return
        except:
            pass

        try:
            # Linux with ALSA
            if sys.platform.startswith("linux"):
                os.system('speaker-test -t sine -f 1000 -l 1 -s 1 2>/dev/null &')
                return
        except:
            pass

        # Fallback: terminal bell
        print('\a', end='', flush=True)


# ==============================
# IMPEDANCE MONITORING
# ==============================
class ImpedanceMonitor:
    def __init__(self, board: BoardShim, threshold: float = Config.IMPEDANCE_THRESHOLD):
        self.board = board
        self.threshold = threshold
        self.impedance_log = []

    def check_impedance(self) -> Dict[int, float]:
        """Check impedance for all channels. Returns dict of channel: impedance values."""
        eeg_channels = BoardShim.get_eeg_channels(self.board.get_board_id())
        impedances = {}

        # Get short data sample to estimate impedance
        try:
            data = self.board.get_current_board_data(Config.SAMPLING_RATE)  # 1 second
        except:
            # If no data available, mark all as poor
            for ch in eeg_channels:
                impedances[ch] = float('inf')
            return impedances

        for ch in eeg_channels:
            if data.shape[1] > 0:
                # Simple impedance estimation based on signal variance
                # High impedance = high noise/variance in DC-coupled systems
                channel_data = data[ch]
                impedance_estimate = np.var(channel_data) * 1000  # rough estimation
                impedances[ch] = impedance_estimate
            else:
                impedances[ch] = float('inf')

        self.log_impedance(impedances)
        return impedances

    def log_impedance(self, impedances: Dict[int, float]):
        """Log impedance values with timestamp."""
        timestamp = datetime.now().isoformat()
        for ch, imp in impedances.items():
            self.impedance_log.append({
                'timestamp': timestamp,
                'channel': ch,
                'impedance': imp,
                'status': 'good' if imp < self.threshold else 'poor'
            })

    def get_good_channels(self) -> List[int]:
        """Return list of channels with good impedance."""
        impedances = self.check_impedance()
        good_channels = [ch for ch, imp in impedances.items() if imp < self.threshold]

        # If no channels are "good", use all available channels as fallback
        if not good_channels:
            print("WARNING: No channels with good impedance found. Using all channels.")
            good_channels = list(impedances.keys())

        return good_channels

    def save_impedance_log(self):
        """Save impedance log to CSV."""
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        if self.impedance_log:
            df = pd.DataFrame(self.impedance_log)
            filepath = os.path.join(Config.DATA_DIR, Config.IMPEDANCE_LOG)
            df.to_csv(filepath, index=False)
            print(f"Impedance log saved to {filepath}")


# ==============================
# ENHANCED STIMULUS WINDOW
# ==============================
class EnhancedColorStimulus(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Red/Blue Visual Stimulus")
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint)
        self.showFullScreen()
        self.audio = AudioFeedback()

        # Add instruction label
        self.instruction_label = QtWidgets.QLabel(self)
        self.instruction_label.setAlignment(QtCore.Qt.AlignCenter)
        self.instruction_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: white;
                background-color: rgba(0, 0, 0, 100);
                padding: 20px;
                border-radius: 10px;
            }
        """)
        self.instruction_label.hide()

    def resizeEvent(self, event):
        """Center the instruction label when window is resized."""
        super().resizeEvent(event)
        if hasattr(self, 'instruction_label'):
            self.instruction_label.resize(400, 100)
            center = self.rect().center()
            self.instruction_label.move(center.x() - 200, center.y() - 50)

    def show_instruction(self, text: str):
        """Show instruction text."""
        self.instruction_label.setText(text)
        self.instruction_label.show()
        self.repaint()
        QtWidgets.QApplication.processEvents()

    def hide_instruction(self):
        """Hide instruction text."""
        self.instruction_label.hide()
        self.repaint()
        QtWidgets.QApplication.processEvents()

    def show_color(self, color: str):
        """Display color stimulus."""
        self.hide_instruction()

        if color.lower() == 'red':
            bg_color = QtGui.QColor(255, 0, 0)  # Pure red
        elif color.lower() == 'blue':
            bg_color = QtGui.QColor(0, 0, 255)  # Pure blue
        elif color.lower() == 'black':
            bg_color = QtGui.QColor(0, 0, 0)  # Black for rest
        else:
            bg_color = QtGui.QColor(128, 128, 128)  # Gray for preparation

        palette = self.palette()
        palette.setColor(self.backgroundRole(), bg_color)
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self.repaint()
        QtWidgets.QApplication.processEvents()

    def show_fixation_cross(self):
        """Show fixation cross during preparation."""
        self.show_color('gray')
        self.show_instruction("+")

    def play_beep(self):
        """Play audio feedback."""
        self.audio.play_beep()

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()


# ==============================
# ADVANCED FEATURE EXTRACTION
# ==============================
class AdvancedFeatureExtractor:
    def __init__(self, sampling_rate: int = Config.SAMPLING_RATE):
        self.sampling_rate = sampling_rate
        self.window_size = int(Config.WINDOW_SIZE * sampling_rate)

    def preprocess_data(self, data: np.ndarray, channels: List[int]) -> np.ndarray:
        """Apply comprehensive preprocessing to EEG data."""
        processed_data = data.copy()

        for ch in channels:
            if ch < processed_data.shape[0]:  # Check if channel exists
                channel_data = processed_data[ch]

                # Skip if channel has no data
                if len(channel_data) == 0:
                    continue

                try:
                    # Remove DC offset
                    DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)

                    # Band-pass filter (0.5-40 Hz)
                    DataFilter.perform_bandpass(
                        channel_data, self.sampling_rate,
                        Config.FILTER_LOW, Config.FILTER_HIGH, 4,
                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
                    )

                    # Notch filter (remove powerline noise)
                    DataFilter.perform_bandstop(
                        channel_data, self.sampling_rate,
                        Config.NOTCH_FREQ - 2, Config.NOTCH_FREQ + 2, 4,
                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0
                    )

                    # Optional: Remove environmental noise
                    DataFilter.remove_environmental_noise(
                        channel_data, self.sampling_rate, NoiseTypes.SIXTY.value
                    )
                except Exception as e:
                    print(f"Warning: Failed to preprocess channel {ch}: {e}")

        return processed_data

    def extract_comprehensive_features(self, data: np.ndarray, channels: List[int]) -> np.ndarray:
        """Extract comprehensive features for visual classification."""
        features = []

        for ch in channels:
            if ch >= data.shape[0]:
                # Channel doesn't exist, add zeros
                features.extend([0.0] * 18)  # 18 features per channel
                continue

            channel_data = data[ch]

            if len(channel_data) == 0:
                features.extend([0.0] * 18)
                continue

            try:
                # Time domain features
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.max(channel_data) - np.min(channel_data),  # Peak-to-peak
                    np.var(channel_data),
                    np.median(channel_data),
                    np.percentile(channel_data, 75) - np.percentile(channel_data, 25)  # IQR
                ])

                # Frequency domain features
                try:
                    psd = DataFilter.get_psd_welch(
                        channel_data,
                        DataFilter.get_nearest_power_of_two(self.sampling_rate),
                        self.sampling_rate // 2,
                        self.sampling_rate,
                        WindowOperations.BLACKMAN_HARRIS.value
                    )

                    # Standard EEG bands
                    delta_power = DataFilter.get_band_power(psd, 0.5, 4.0)
                    theta_power = DataFilter.get_band_power(psd, 4.0, 8.0)
                    alpha_power = DataFilter.get_band_power(psd, 8.0, 12.0)
                    beta_power = DataFilter.get_band_power(psd, 12.0, 30.0)
                    gamma_power = DataFilter.get_band_power(psd, 30.0, 40.0)

                    total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power

                    # Relative band powers
                    features.extend([
                        delta_power / total_power if total_power > 0 else 0,
                        theta_power / total_power if total_power > 0 else 0,
                        alpha_power / total_power if total_power > 0 else 0,
                        beta_power / total_power if total_power > 0 else 0,
                        gamma_power / total_power if total_power > 0 else 0
                    ])

                    # Band ratios (important for visual processing)
                    features.extend([
                        alpha_power / beta_power if beta_power > 0 else 0,
                        theta_power / alpha_power if alpha_power > 0 else 0,
                        beta_power / gamma_power if gamma_power > 0 else 0
                    ])

                    # Visual processing specific bands
                    occipital_alpha = DataFilter.get_band_power(psd, 8.0, 13.0)  # Extended alpha
                    visual_gamma = DataFilter.get_band_power(psd, 30.0, 45.0)  # High gamma

                    features.extend([
                        occipital_alpha / total_power if total_power > 0 else 0,
                        visual_gamma / total_power if total_power > 0 else 0
                    ])

                except Exception as e:
                    print(f"Warning: Failed to compute PSD for channel {ch}: {e}")
                    # Add zeros for frequency features
                    features.extend([0.0] * 12)

            except Exception as e:
                print(f"Warning: Failed to extract features for channel {ch}: {e}")
                features.extend([0.0] * 18)

        return np.array(features)


# ==============================
# DATA COLLECTION MANAGER
# ==============================
class DataCollectionManager:
    def __init__(self, board: BoardShim, subject_id: str):
        self.board = board
        self.subject_id = subject_id
        self.impedance_monitor = ImpedanceMonitor(board)
        self.feature_extractor = AdvancedFeatureExtractor()

        # Create directories
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True)

    def collect_session_data(self, session_name: str = None) -> pd.DataFrame:
        """Collect a complete session of red/blue data."""
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"\n=== Starting Data Collection Session: {session_name} ===")

        # Setup stimulus window
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        stimulus = EnhancedColorStimulus()

        try:
            # Check impedance before starting
            print("Checking electrode impedance...")
            good_channels = self.impedance_monitor.get_good_channels()
            print(f"Channels with good impedance: {good_channels}")

            if len(good_channels) < 2:
                print("WARNING: Less than 2 channels have good impedance!")
                stimulus.show_instruction("Check electrode connections\nPress any key to continue anyway")
                app.processEvents()
                input("Press Enter to continue...")

            # Start data collection
            self.board.start_stream()
            time.sleep(1)  # Let stream stabilize

            dataset = []

            # Create trial sequence
            colors = [("red", 0), ("blue", 1)]
            trials = colors * Config.TRIALS_PER_COLOR
            random.shuffle(trials)

            print(f"Collecting {len(trials)} trials...")
            stimulus.show_instruction(f"Starting {len(trials)} trials\nPress ESC to abort")
            time.sleep(3)

            for trial_idx, (color_name, label) in enumerate(trials):
                print(f"\nTrial {trial_idx + 1}/{len(trials)}: {color_name}")

                # Preparation phase
                stimulus.show_instruction(f"Trial {trial_idx + 1}\nGet ready...")
                time.sleep(1)
                stimulus.show_fixation_cross()
                time.sleep(Config.PREPARATION_TIME)

                # Stimulus phase
                stimulus.show_color(color_name)
                stimulus.play_beep()

                # Mark stimulus onset
                try:
                    self.board.insert_marker(label + 1)  # 1 for red, 2 for blue
                except:
                    pass  # Continue even if marker insertion fails

                # Collect data during stimulus
                time.sleep(Config.TRIAL_DURATION)

                # Get trial data
                num_samples = int(Config.TRIAL_DURATION * Config.SAMPLING_RATE)
                try:
                    trial_data = self.board.get_current_board_data(num_samples)
                except:
                    trial_data = self.board.get_board_data()

                if trial_data.shape[1] > 0:
                    # Process and extract features
                    processed_data = self.feature_extractor.preprocess_data(trial_data, good_channels)
                    features = self.feature_extractor.extract_comprehensive_features(
                        processed_data, good_channels
                    )

                    # Store trial info
                    trial_info = {
                        'subject_id': self.subject_id,
                        'session_name': session_name,
                        'trial_number': trial_idx + 1,
                        'color': color_name,
                        'label': label,
                        'timestamp': datetime.now().isoformat(),
                        'channels_used': str(good_channels),
                        'n_features': len(features)
                    }

                    # Add features as separate columns
                    for i, feat in enumerate(features):
                        trial_info[f'feature_{i:03d}'] = feat

                    dataset.append(trial_info)
                else:
                    print(f"Warning: No data collected for trial {trial_idx + 1}")

                # Rest phase
                stimulus.show_color("black")
                time.sleep(Config.REST_DURATION)

                # Periodic impedance check
                if (trial_idx + 1) % 10 == 0:
                    print("Checking impedance...")
                    current_good = self.impedance_monitor.get_good_channels()
                    if len(current_good) < len(good_channels):
                        print("WARNING: Impedance may have degraded during session!")

            self.board.stop_stream()

            # Save impedance log
            self.impedance_monitor.save_impedance_log()

            # Create DataFrame and save
            if dataset:
                df = pd.DataFrame(dataset)
                filename = f"{self.subject_id}_{session_name}.csv"
                filepath = os.path.join(Config.DATA_DIR, filename)
                df.to_csv(filepath, index=False)

                print(f"\nSession data saved to: {filepath}")
                print(f"Collected {len(df)} trials")

                stimulus.show_instruction(f"Session Complete!\n{len(df)} trials collected\nData saved to {filename}")
                time.sleep(3)

                return df
            else:
                print("No data collected!")
                stimulus.show_instruction("No data collected!\nCheck your setup")
                time.sleep(3)
                return None

        except Exception as e:
            print(f"Error during data collection: {e}")
            stimulus.show_instruction(f"Error: {e}\nSession aborted")
            time.sleep(3)
            return None
        finally:
            stimulus.close()


# ==============================
# MODEL TRAINING MANAGER
# ==============================
class ModelTrainingManager:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=Config.RANDOM_STATE,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=6, random_state=Config.RANDOM_STATE
            ),
            'svm': SVC(
                kernel='rbf', C=1.0, gamma='scale', random_state=Config.RANDOM_STATE,
                probability=True
            ),
            'logistic_regression': LogisticRegression(
                C=1.0, random_state=Config.RANDOM_STATE, max_iter=1000
            )
        }

    def prepare_data(self, dataframes: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Prepare data for training from multiple DataFrames."""
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Get feature columns
        feature_cols = [col for col in combined_df.columns if col.startswith('feature_')]

        if not feature_cols:
            raise ValueError("No feature columns found in data!")

        X = combined_df[feature_cols].values
        y = combined_df['label'].values

        # Remove any NaN or infinite values
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        combined_df = combined_df[mask].reset_index(drop=True)

        print(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: Red={np.sum(y == 0)}, Blue={np.sum(y == 1)}")

        return X, y, combined_df

    def train_and_evaluate_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train and evaluate multiple models."""
        results = {}

        # Check if we have enough data
        if len(X) < 10:
            raise ValueError("Not enough data for training! Need at least 10 samples.")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=min(Config.TEST_SIZE, 0.4),  # Use smaller test set if data is limited
            random_state=Config.RANDOM_STATE,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        for name, base_model in self.models.items():
            print(f"\n--- Training {name} ---")

            try:
                # Create pipeline with scaling
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', base_model)
                ])

                # Cross-validation (use fewer folds if data is limited)
                cv_folds = min(Config.CV_FOLDS, len(X_train) // 2, 3)
                if cv_folds >= 2:
                    cv_scores = cross_val_score(
                        pipeline, X_train, y_train,
                        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                           random_state=Config.RANDOM_STATE),
                        scoring='accuracy'
                    )
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                else:
                    cv_mean = cv_std = 0.0
                    print("Not enough data for cross-validation")

                # Train on full training set
                pipeline.fit(X_train, y_train)

                # Test set evaluation
                y_pred = pipeline.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)

                # Store results
                results[name] = {
                    'model': pipeline,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'test_accuracy': test_accuracy,
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'classification_report': classification_report(y_test, y_pred, zero_division=0)
                }

                print(f"CV Accuracy: {cv_mean:.3f} (+/- {cv_std * 2:.3f})")
                print(f"Test Accuracy: {test_accuracy:.3f}")

            except Exception as e:
                print(f"Error training {name}: {e}")
                continue

        return results

    def save_best_model(self, results: Dict, subject_id: str = "general") -> str:
        """Save the best performing model."""
        if not results:
            raise ValueError("No models were successfully trained!")

        # Find best model based on test accuracy (more reliable than CV for small datasets)
        best_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
        best_model = results[best_name]['model']

        # Model metadata
        model_info = {
            'model': best_model,
            'model_type': best_name,
            'cv_accuracy': results[best_name]['cv_mean'],
            'test_accuracy': results[best_name]['test_accuracy'],
            'subject_id': subject_id,
            'training_date': datetime.now().isoformat(),
            'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('_')}
        }

        # Save model
        filename = f"{subject_id}_red_blue_model.pkl"
        filepath = os.path.join(Config.MODEL_DIR, filename)
        joblib.dump(model_info, filepath)

        print(f"\nBest model ({best_name}) saved to: {filepath}")
        print(f"CV Accuracy: {results[best_name]['cv_mean']:.3f}")
        print(f"Test Accuracy: {results[best_name]['test_accuracy']:.3f}")

        return filepath


# ==============================
# REAL-TIME PREDICTION SYSTEM
# ==============================
class RealTimePredictionSystem:
    def __init__(self, model_path: str, board: BoardShim):
        self.board = board
        self.model_info = joblib.load(model_path)
        self.model = self.model_info['model']
        self.feature_extractor = AdvancedFeatureExtractor()

        # Use all EEG channels for prediction
        self.channels = BoardShim.get_eeg_channels(board.get_board_id())

        print(f"Loaded model: {self.model_info['model_type']}")
        print(f"Training accuracy: {self.model_info['test_accuracy']:.3f}")
        print(f"Using channels: {self.channels}")

    def predict_from_current_data(self) -> Tuple[str, float]:
        """Get prediction from current EEG data."""
        # Get recent data
        window_samples = int(Config.WINDOW_SIZE * Config.SAMPLING_RATE)

        try:
            data = self.board.get_current_board_data(window_samples)
        except:
            return "no_data", 0.0

        if data.shape[1] < window_samples // 2:  # Need at least half the window
            return "insufficient_data", 0.0

        # Use the most recent available data
        recent_data = data[:, -min(window_samples, data.shape[1]):]

        try:
            # Preprocess and extract features
            processed_data = self.feature_extractor.preprocess_data(recent_data, self.channels)
            features = self.feature_extractor.extract_comprehensive_features(
                processed_data, self.channels
            )

            # Make prediction
            prediction = self.model.predict(features.reshape(1, -1))[0]

            try:
                confidence = self.model.predict_proba(features.reshape(1, -1)).max()
            except:
                confidence = 0.5  # Default confidence if probability not available

            color = "" if prediction == 0 else "blue"
            return color, confidence

        except Exception as e:
            print(f"Prediction error: {e}")
            return "error", 0.0

    def run_real_time_demo(self):
        """Run real-time prediction with visual feedback."""
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        stimulus = EnhancedColorStimulus()  # Fixed: Added missing parentheses

        self.board.start_stream()
        time.sleep(2)  # Let stream stabilize

        print("Real-time prediction started. Press ESC to stop.")
        print("The screen will show the predicted color...")

        stimulus.show_instruction("Real-time Prediction\nPress ESC to stop")
        time.sleep(2)

        try:
            prediction_count = []
            sample_lim = 10
            while True:
                prediction, confidence = self.predict_from_current_data()
                prediction_count.append(prediction)
                if len(prediction_count) < sample_lim:
                    prediction_count.pop(-1)
                avg_pred = "blue" if prediction_count.count("blue") > prediction_count.count("red") else "red"
                if prediction not in ["insufficient_data", "no_data", "error"]:
                    stimulus.show_color(prediction)
                    print(f"Prediction: {prediction} (confidence: {confidence:.2f})")
                elif prediction == "insufficient_data":
                    stimulus.show_color("gray")
                    print("Insufficient data for prediction")
                elif prediction == "no_data":
                    stimulus.show_color("black")
                    print("No data available")
                else:  # error
                    stimulus.show_color("gray")
                    print("Prediction error")

                time.sleep(0.5)  # Update every 500ms

                # Process Qt events to check for ESC key
                app.processEvents()

        except KeyboardInterrupt:
            print("\nStopping real-time prediction...")
        finally:
            self.board.stop_stream()
            stimulus.close()


# ==============================
# MAIN EXECUTION FUNCTIONS
# ==============================
def collect_data_session():
    """Main function for data collection."""
    parser = argparse.ArgumentParser(description="Collect EEG data for red/blue classification")
    parser.add_argument('--subject-id', type=str, required=True,
                        help='Subject identifier (e.g., subject01)')
    parser.add_argument('--session-name', type=str,
                        help='Session name (optional)')
    args = parser.parse_args()

    # Setup board
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = Config.SERIAL_PORT

    board = BoardShim(Config.BOARD_ID, params)

    try:
        board.prepare_session()

        # Create collection manager
        collector = DataCollectionManager(board, args.subject_id)

        # Collect data
        #IM LAZY SO I WILL TEMPORARILY JUST RENAME SESSION NAMES
        df = collector.collect_session_data(args.session_name)

        if df is not None:
            print(f"\nData collection completed successfully!")
            print(f"Collected {len(df)} trials")

    finally:
        if board.is_prepared():
            board.release_session()


def train_models():
    """Main function for model training."""
    parser = argparse.ArgumentParser(description="Train models for red/blue classification")
    parser.add_argument('--data-dir', type=str, default=Config.DATA_DIR,
                        help='Directory containing training data')
    parser.add_argument('--subject-id', type=str, default='general',
                        help='Subject ID for model (use "general" for multi-subject)')
    args = parser.parse_args()

    # Load all CSV files
    csv_files = [f for f in os.listdir(args.data_dir) if f.endswith('.csv')
                 and not f.startswith('impedance')]

    if not csv_files:
        print(f"No data files found in {args.data_dir}")
        return

    print(f"Found {len(csv_files)} data files")

    dataframes = []
    for csv_file in csv_files:
        try:
            filepath = os.path.join(args.data_dir, csv_file)
            df = pd.read_csv(filepath)
            dataframes.append(df)
            print(f"Loaded {csv_file}: {len(df)} trials")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    if not dataframes:
        print("No valid data files found!")
        return

    # Train models
    try:
        trainer = ModelTrainingManager()
        X, y, combined_df = trainer.prepare_data(dataframes)
        results = trainer.train_and_evaluate_models(X, y)

        if results:
            # Print results
            print("\n=== MODEL COMPARISON ===")
            for name, result in results.items():
                print(f"{name:20} | CV: {result['cv_mean']:.3f} | Test: {result['test_accuracy']:.3f}")

            # Save best model
            trainer.save_best_model(results, args.subject_id)
        else:
            print("No models were successfully trained!")

    except Exception as e:
        print(f"Error during training: {e}")


def run_real_time_prediction():
    """Main function for real-time prediction."""
    parser = argparse.ArgumentParser(description="Run real-time red/blue prediction")
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model file')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        return

    # Setup board
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = Config.SERIAL_PORT

    board = BoardShim(Config.BOARD_ID, params)

    try:
        board.prepare_session()

        # Create prediction system
        predictor = RealTimePredictionSystem(args.model_path, board)

        # Run real-time demo
        predictor.run_real_time_demo()

    except Exception as e:
        print(f"Error during real-time prediction: {e}")
    finally:
        if board.is_prepared():
            board.release_session()


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python red_blue_classifier.py collect --subject-id SUBJECT_ID [--session-name NAME]")
        print("  python red_blue_classifier.py train [--data-dir DIR] [--subject-id ID]")
        print("  python red_blue_classifier.py predict --model-path MODEL_FILE")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = sys.argv[1:]  # Remove the command from argv

    if command == "collect":
        collect_data_session()
    elif command == "train":
        train_models()
    elif command == "predict":
        run_real_time_prediction()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
#note CHANGE SUBJECT ID TO NAMES AND SESSION NAMES FOR ADDITION OF CSV


# PREDICT: python3 red_blue_classifier.py predict --model-path models/subject01_red_blue_model.pkl
# PREDICT: python3 red_blue_classifier.py predict --model-path models/subjectSantos_red_blue_model.pkl
# python3 red_blue_classifier.py predict --model-path models/subjectMinh_red_blue_model.pkl
# python3 red_blue_classifier.py predict --model-path models/subjectChanyoo_red_blue_model.pkl
# PREDICT: python3 red_blue_classifier.py predict --model-path models/general_red_blue_model.pkl
# PREDICT: python3 red_blue_classifier.py predict --model-path models/subjectNam_red_blue_model.pkl

# Train models using all collected data
#python3 red_blue_classifier.py train --subject-id subject01
#python3 red_blue_classifier.py train --subject-id subjectSantos
#python3 red_blue_classifier.py train --subject-id subjectMinh
#python3 red_blue_classifier.py train --subject-id subjectChanyoo
#python3 red_blue_classifier.py train --subject-id subjectNam

# Train a general model using data from multiple subjects
#python3 red_blue_classifier.py train --subject-id general

# Collect training data for a subject
#python3 red_blue_classifier.py collect --subject-id subject01 --session-name morning_session
#python3 red_blue_classifier.py collect --subject-id subjectSantos --session-name santos_session_1
#python3 red_blue_classifier.py collect --subject-id subjectMinh --session-name minh_session_1
#python3 red_blue_classifier.py collect --subject-id subjectChanyoo --session-name chanyoo_session_1
#python3 red_blue_classifier.py collect --subject-id subjectNam --session-name nam_session_1

# Collect multiple sessions for better training data
#python3 red_blue_classifier.py collect --subject-id subject01 --session-name afternoon_session

#Im still skeptical of the impedance check
