import time
import pickle
import numpy as np
import pandas as pd
import logging
from collections import deque

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import (
    DataFilter,
    WindowOperations,
    DetrendOperations,
    FilterTypes,
)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from pyqtgraph.Qt import QtWidgets, QtGui


# ==============================
# CONFIG
# ==============================
SERIAL_PORT = "/dev/cu.usbserial-DP04VYIJ"
BOARD_ID = BoardIds.SYNTHETIC_BOARD.value
DATA_FILE = "color_eeg_dataset.csv"
MODEL_FILE = "color_bci_model.pkl"

PREDICTION_WINDOW = 2
SMOOTHING_WINDOW = 5
ARTIFACT_THRESHOLD = 150  # microvolts


# ==============================
# GUI
# ==============================
class PredictionWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI Prediction")
        self.showFullScreen()

    def show_color(self, color):
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QtGui.QColor(color))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self.repaint()
        QtWidgets.QApplication.processEvents()


# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_band_features(data, sampling_rate, channels):
    """
    Returns log-relative band powers averaged across channels.
    Bands: delta, theta, alpha, beta, gamma
    """

    band_features = np.zeros(5)

    for ch in channels:
        channel_data = data[ch].copy()


        # ----------------------------
        # Detrend
        # ----------------------------
        DataFilter.detrend(
            channel_data,
            DetrendOperations.CONSTANT.value
        )

        # ----------------------------
        # Bandpass 1–45 Hz
        # ----------------------------
        DataFilter.perform_bandpass(
            channel_data,
            sampling_rate,
            3.0,
            45.0,
            4,
            FilterTypes.BUTTERWORTH_ZERO_PHASE,
            0,
        )

        # ----------------------------
        # Notch 50 Hz
        # ----------------------------
        DataFilter.perform_bandstop(
            channel_data,
            sampling_rate,
            50.0,
            2.0,
            4,
            FilterTypes.BUTTERWORTH_ZERO_PHASE,
            0,
        )

        # ----------------------------
        # Notch 60 Hz
        # ----------------------------
        DataFilter.perform_bandstop(
            channel_data,
            sampling_rate,
            60.0,
            2.0,
            4,
            FilterTypes.BUTTERWORTH_ZERO_PHASE,
            0,
        )

        # ----------------------------
        # PSD (Welch)
        # ----------------------------
        nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

        psd = DataFilter.get_psd_welch(
            channel_data,
            nfft,
            nfft // 2,
            sampling_rate,
            WindowOperations.BLACKMAN_HARRIS.value,
        )

        # ----------------------------
        # Band powers
        # ----------------------------
        delta = DataFilter.get_band_power(psd, 0.5, 4.0)
        theta = DataFilter.get_band_power(psd, 4.0, 8.0)
        alpha = DataFilter.get_band_power(psd, 8.0, 12.0)
        beta  = DataFilter.get_band_power(psd, 12.0, 30.0)
        gamma = DataFilter.get_band_power(psd, 30.0, 45.0)

        total_power = delta + theta + alpha + beta + gamma + 1e-8

        # ----------------------------
        # Relative power (important)
        # ----------------------------
        band_features += np.array([
            delta / total_power,
            theta / total_power,
            alpha / total_power,
            beta  / total_power,
            gamma / total_power,
        ])

    # Average across channels
    band_features /= len(channels)

    # ----------------------------
    # Log transform (VERY important)
    # ----------------------------
    band_features = np.log10(band_features + 1e-8)

    return band_features


# ==============================
# TRAIN MODEL
# ==============================
def train_model():
    df = pd.read_csv(DATA_FILE)

    X = df[["delta", "theta", "alpha", "beta", "gamma"]]
    y = df["label"]

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=2,
            gamma="scale",
            probability=True
        )),
    ])

    scores = cross_val_score(clf, X, y, cv=5)
    print("Cross-validation accuracy:", scores.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print("Final test accuracy:", acc)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)

    print("Model saved.")

    return clf


# ==============================
# LIVE PREDICTION
# ==============================
def live_prediction(clf):
    logging.basicConfig(level=logging.INFO)
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT

    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    board.start_stream()

    sampling_rate = 250  # your board
    channels = BoardShim.get_exg_channels(BOARD_ID)

    app = QtWidgets.QApplication([])
    gui = PredictionWindow()

    prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

    print("Realtime prediction started. Ctrl+C to stop.")

    try:
        while True:
            time.sleep(PREDICTION_WINDOW)

            num_samples = int(PREDICTION_WINDOW * sampling_rate)
            data = board.get_current_board_data(num_samples)

            if data.shape[1] < num_samples:
                continue

            features = extract_band_features(
                data,
                sampling_rate,
                channels
            )

            if features is None:
                print("Artifact detected. Skipping.")
                continue

            features = features.reshape(1, -1)

            prediction = clf.predict(features)[0]
            prob = np.max(clf.predict_proba(features))

            prediction_buffer.append(prediction)

            # ----------------------------
            # Majority vote smoothing
            # ----------------------------
            if len(prediction_buffer) == SMOOTHING_WINDOW:
                smoothed = max(
                    set(prediction_buffer),
                    key=prediction_buffer.count
                )
            else:
                smoothed = prediction

            if smoothed == 0:
                gui.show_color("red")
                print(f"🔴 RED  | confidence: {prob:.2f}")
            else:
                gui.show_color("blue")
                print(f"🔵 BLUE | confidence: {prob:.2f}")

    except KeyboardInterrupt:
        print("Stopping...")
        board.stop_stream()
        board.release_session()
        gui.close()



# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    clf = train_model()
    live_prediction(clf)
