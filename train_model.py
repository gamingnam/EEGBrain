# train_model.py
import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes
from utils import speak, get_timestamp, save_csv_with_header

SAVE_DIR = "models/training_data"
MODEL_DIR = "models/trained_models"

def train_new_model():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    speak("Starting new training session.")
    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DP04VYIJ"
    board = BoardShim(BoardIds.CYTON_BOARD.value, params)

    board.prepare_session()
    board.start_stream()

    rtime = 3
    sfreq = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
    speak("Recording five rounds of eye open and eye closed.")
    X, y = [], []

    for i in range(10):
        # Eyes open
        speak(f"Round {i + 1}: Please keep your eyes open.")
        time.sleep(1) #to give time to change
        speak("Go.")
        time.sleep(rtime)  # short buffer for user to react
        data_open = board.get_current_board_data(sfreq*rtime)  # ~10s at 250Hz
        X.append(data_open[eeg_channels, :])
        y.extend([0] * data_open.shape[1])
        save_csv_with_header(data_open, f"{SAVE_DIR}/open_{get_timestamp()}.csv")

        # Eyes closed
        speak("Now close your eyes.")
        time.sleep(1)  # to give time to change
        speak("Go.")
        time.sleep(rtime)
        data_closed = board.get_current_board_data(sfreq*rtime)
        X.append(data_closed[eeg_channels, :])
        y.extend([1] * data_closed.shape[1])
        save_csv_with_header(data_closed, f"{SAVE_DIR}/closed_{get_timestamp()}.csv")

    board.stop_stream()
    board.release_session()

    # Convert to features
    X = np.concatenate(X, axis=1)
    features = []
    for row in X:
        DataFilter.perform_bandpass(row, sfreq, 8.0, 12.0, 4, FilterTypes.BUTTERWORTH.value, 0)
        features.append(np.mean(row ** 2))
    features = np.array(features).reshape(-1, 1)
    y = np.array(y[:features.shape[0]])

    clf = LDA()
    clf.fit(features, y)

    model_name = f"{MODEL_DIR}/lda_{get_timestamp()}.pkl"
    with open(model_name, "wb") as f:
        pickle.dump(clf, f)

    speak(f"Training complete. Model saved as {model_name}")
    return model_name
