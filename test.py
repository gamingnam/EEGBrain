import time
import random
import numpy as np
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import os


from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations, DetrendOperations, FilterTypes
from pyqtgraph.Qt import QtWidgets, QtGui, QtCore


# ==============================
# CONFIG
# ==============================
SERIAL_PORT = "/dev/cu.usbserial-DP04VYIJ"
#SYNTHETIC_BOARD
BOARD_ID = BoardIds.CYTON_BOARD.value
TRIAL_DURATION = 3.0        # seconds per color
REST_DURATION = 2.0         # seconds between trials
TRIALS_PER_COLOR = 20
SAVE_FILE = "color_eeg_dataset.csv"
TRAIN_FILE = "color_eeg_train.csv"
TEST_FILE = "color_eeg_test.csv"



# ==============================
# STIMULUS WINDOW
# ==============================
class ColorStimulus(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Color Stimulus")
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
    features = np.zeros(5)

    for ch in channels:
        channel_data = data[ch]

        DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(data[ch], sampling_rate, 3.0, 45.0, 2,
                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(data[ch], sampling_rate, 48.0, 52.0, 2,
                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(data[ch], sampling_rate, 58.0, 62.0, 2,
                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)

        psd = DataFilter.get_psd_welch(
            channel_data,
            DataFilter.get_nearest_power_of_two(sampling_rate),
            sampling_rate // 2,
            sampling_rate,
            WindowOperations.BLACKMAN_HARRIS.value
        )

        features[0] += DataFilter.get_band_power(psd, 0.5, 4.0)   # Delta
        features[1] += DataFilter.get_band_power(psd, 4.0, 8.0)   # Theta
        features[2] += DataFilter.get_band_power(psd, 8.0, 12.0)  # Alpha
        features[3] += DataFilter.get_band_power(psd, 12.0, 30.0) # Beta
        features[4] += DataFilter.get_band_power(psd, 30.0, 45.0) # Gamma

    return features / len(channels)


# ==============================
# MAIN
# ==============================
def main():
    logging.basicConfig(level=logging.INFO)
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT

    board = BoardShim(BOARD_ID, params)
    board.prepare_session()
    board.start_stream()

    sampling_rate = BoardShim.get_sampling_rate(BOARD_ID)
    channels = BoardShim.get_exg_channels(BOARD_ID)

    app = QtWidgets.QApplication([])
    stimulus = ColorStimulus()

    dataset = []

    colors = [("red", 0), ("blue", 1)]
    trials = colors * TRIALS_PER_COLOR
    random.shuffle(trials)

    for color_name, label in trials:
        # Show color
        stimulus.show_color(color_name)
        time.sleep(TRIAL_DURATION)

        # Collect data
        num_samples = int(TRIAL_DURATION * sampling_rate)
        data = board.get_current_board_data(num_samples)

        features = extract_band_features(data, sampling_rate, channels)
        trial_number = len(dataset) + 1
        timestamp = time.time()

        row = np.concatenate((features, [label, trial_number, timestamp]))
        dataset.append(row)

        # Rest (black screen)
        stimulus.show_color("black")
        time.sleep(REST_DURATION)

        print(f"Recorded {color_name}")

    dataset = np.array(dataset)

    columns = [
        "delta",
        "theta",
        "alpha",
        "beta",
        "gamma",
        "label",
        "trial_number",
        "timestamp"
    ]

    df = pd.DataFrame(dataset, columns=columns)

    # -----------------------------
    # Append or Create CSV
    # -----------------------------
    if os.path.exists(SAVE_FILE):
        df.to_csv(SAVE_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(SAVE_FILE, index=False)

    print(f"Saved full dataset to {SAVE_FILE}")

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    X = df[["delta", "theta", "alpha", "beta", "gamma"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print(f"Saved train set to {TRAIN_FILE}")
    print(f"Saved test set to {TEST_FILE}")

    board.stop_stream()

    board.release_session()
    stimulus.close()


if __name__ == "__main__":
    main()
