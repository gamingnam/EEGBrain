import os
import sys
import time
import numpy as np
import joblib

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# ==========================
# Global config
# ==========================

# CHANGE THIS to your Cyton serial port
SERIAL_PORT = '/dev/cu.usbserial-DP04VYIJ'

# Use Cyton 8-ch board; change if needed
DEFAULT_BOARD_ID = BoardIds.CYTON_BOARD.value

DEFAULT_MODEL_PATH = 'eye_state_model.pkl'

# Directory where CSVs will be saved/loaded
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)


# ==========================
# Helper: sound / beep
# ==========================

def play_beep():
    """
    Try a few different ways to make a sound.
    Customize this function for your OS if needed.
    """
    # Windows
    try:
        import winsound
        winsound.Beep(1000, 300)  # 1000 Hz, 300 ms
        return
    except Exception:
        pass

    # macOS: use built-in sounds
    if sys.platform == 'darwin':
        try:
            os.system('afplay /System/Library/Sounds/Pop.aiff')
            return
        except Exception:
            pass

    # Fallback: terminal bell (might not be audible)
    print('\a', end='', flush=True)


# ==========================
# Helper: feature extraction
# ==========================

def extract_features_from_window(eeg_window: np.ndarray) -> np.ndarray:
    """
    eeg_window: shape (n_channels, n_samples)

    Simple features per channel:
      - mean
      - std
      - peak-to-peak (max - min)

    Returns a 1D numpy array of length 3 * n_channels.
    """
    means = eeg_window.mean(axis=1)
    stds = eeg_window.std(axis=1)
    ptp = eeg_window.max(axis=1) - eeg_window.min(axis=1)

    features = np.concatenate([means, stds, ptp], axis=0)
    return features


def extract_features_from_recording(
    eeg_data: np.ndarray,
    sampling_rate: int,
    window_sec: float,
    label: int
):
    """
    eeg_data: shape (n_channels, n_samples) for entire condition
    sampling_rate: Hz
    window_sec: window length in seconds
    label: integer label for all windows from this recording
           (1 = open, 0 = closed)

    Returns:
        X: (n_windows, n_features)
        y: (n_windows,)
    """
    window_size = int(window_sec * sampling_rate)
    n_channels, n_samples = eeg_data.shape

    n_windows = n_samples // window_size
    X_list = []
    y_list = []

    for w in range(n_windows):
        start = w * window_size
        end = start + window_size
        window = eeg_data[:, start:end]
        feats = extract_features_from_window(window)
        X_list.append(feats)
        y_list.append(label)

    if not X_list:
        return np.empty((0, 0)), np.empty((0,), dtype=int)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    return X, y


# ==========================
# Helper: latest CSV finder
# ==========================

def get_most_recent_csv(prefix: str, directory: str = DATA_DIR) -> str:
    """
    Returns the full path of the most recent CSV file in `directory`
    whose filename starts with `prefix` (e.g., 'OPEN_' or 'CLOSED_').
    Filenames are expected to contain a sortable timestamp (YYYYmmdd_HHMMSS).
    """
    files = [
        f for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith('.csv')
    ]
    if not files:
        raise FileNotFoundError(
            f"No CSV files with prefix '{prefix}' found in {directory}"
        )

    # Because we use YYYYMMDD_HHMMSS in the filename, a simple sort works
    files.sort()
    latest = files[-1]
    full_path = os.path.join(directory, latest)
    print(f"Most recent {prefix} file: {full_path}")
    return full_path

def remove_constant_channels(open_data, closed_data, tol=1e-9):
    """
    Remove channels (rows) that are (almost) constant across BOTH open and closed data.
    """
    # Concatenate along time axis
    all_data = np.hstack([open_data, closed_data])
    # Std per channel
    channel_std = all_data.std(axis=1)
    keep = channel_std > tol

    print("Channel stds:", channel_std)
    print("Keeping channels indices:", np.where(keep)[0])

    return open_data[keep, :], closed_data[keep, :], keep

def combine_csv_files_with_prefix(prefix: str, directory: str = DATA_DIR) -> str:
    """
    Combines all CSV files in `directory` whose filenames start with `prefix`
    (e.g., 'OPEN_' or 'CLOSED_') into a single CSV.

    - Assumes each CSV has shape (n_channels, n_samples).
    - Concatenates along the sample axis (horizontally / time axis).
    - Ensures that all files have the same number of channels.

    The output file is named:
        <prefix>COMBINED_YYYYmmdd_HHMMSS.csv

    Returns:
        full path to the combined CSV file.
    """
    os.makedirs(directory, exist_ok=True)

    # Collect all matching CSV filenames
    files = [
        f for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith('.csv')
    ]
    if not files:
        raise FileNotFoundError(
            f"No CSV files with prefix '{prefix}' found in {directory}"
        )

    files.sort()  # Just to have a deterministic order
    print(f"Combining {len(files)} files with prefix '{prefix}' in {directory}:")
    for f in files:
        print("  -", f)


    combined_data_list = []
    reference_n_channels = None

    for fname in files:
        fpath = os.path.join(directory, fname)
        data = np.loadtxt(fpath, delimiter=',')
        print(fname, " - Shape: ", data.shape)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if reference_n_channels is None:
            reference_n_channels = data.shape[0]
        else:
            if data.shape[0] != reference_n_channels:
                raise ValueError(
                    f"File {fpath} has {data.shape[0]} channels, "
                    f"expected {reference_n_channels}."
                )

        combined_data_list.append(data)

    # Concatenate along time / sample axis (columns)
    combined_data = np.hstack(combined_data_list)
    print("Combined data shape:", combined_data.shape)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_name = f"COMBINED_{prefix}_{timestamp}.csv"
    out_path = os.path.join(directory, out_name)

    np.savetxt(out_path, combined_data, delimiter=',')
    print(f"Combined CSV saved to: {out_path}")

    return out_path



# ==========================
# 1) Collect training data
# ==========================

def collect_eye_state_data(
    board: BoardShim,
    board_id: int = DEFAULT_BOARD_ID,
    duration_per_state_sec: int = 60,
    pre_beep_sec: int = 5,
    save_dir: str = DATA_DIR
):
    """
    Collects training data for OPEN and CLOSED eyes using live stream
    and saves them as CSV files:

        OPEN_YYYYmmdd_HHMMSS.csv
        CLOSED_YYYYmmdd_HHMMSS.csv

    Each CSV contains data of shape (n_channels, n_samples).

    Returns:
        open_data:   np.ndarray, shape (n_channels, n_samples_open)
        closed_data: np.ndarray, shape (n_channels, n_samples_closed)
        open_path:   str, path to OPEN CSV
        closed_path: str, path to CLOSED CSV
    """
    os.makedirs(save_dir, exist_ok=True)

    eeg_channels = BoardShim.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"EEG channels: {eeg_channels}")

    def collect_one_condition(condition_name: str):
        print(f"\n===== {condition_name.upper()} EYES =====")
        if condition_name.lower() == "open":
            print("Please OPEN your eyes and look at a fixed point.")
        else:
            print("Please CLOSE your eyes and relax.")
        print(f"Recording will start in {pre_beep_sec} seconds...")
        play_beep()
        time.sleep(pre_beep_sec)

        print(f"Starting {condition_name.upper()} eyes recording for "
              f"{duration_per_state_sec} seconds...")
        board.start_stream()
        time.sleep(duration_per_state_sec)
        play_beep()
        print(f"Stopping {condition_name.upper()} eyes recording.")
        data = board.get_board_data()  # flush internal buffer
        board.stop_stream()

        eeg_data = data[eeg_channels, :]
        print(eeg_data)



        print(f"{condition_name.capitalize()} eyes data shape: {eeg_data.shape}")

        # Save to CSV with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = "OPEN" if condition_name.lower() == "open" else "CLOSED"
        filename = f"{prefix}_{timestamp}.csv"
        filepath = os.path.join(save_dir, filename)

        # Rows = channels, columns = samples
        np.savetxt(filepath, eeg_data, delimiter=',')
        print(f"Saved {condition_name} eyes data to: {filepath}")

        return eeg_data, filepath

    open_data, open_path = collect_one_condition("open")
    closed_data, closed_path = collect_one_condition("closed")

    return open_data, closed_data, open_path, closed_path


# ==========================
# 2) Train + evaluate + save
# ==========================

def train_and_save_eye_state_model(
    open_csv_path: str,
    closed_csv_path: str,
    board_id: int = DEFAULT_BOARD_ID,
    window_sec: float = 1.0,
    model_path: str = DEFAULT_MODEL_PATH
):
    """
    Train a classifier using OPEN and CLOSED eye CSV files.

    CSV format:
      - Saved by collect_eye_state_data
      - Shape: (n_channels, n_samples)

    Labels:
      1 = OPEN eyes
      0 = CLOSED eyes

    Saves a dict via joblib:
        {
            "model": RandomForestClassifier,
            "sampling_rate": int,
            "board_id": int,
            "n_channels": int,
            "window_sec": float
        }

    Prints evaluation stats on both TRAIN and TEST sets.
    """

    print(f"\nLoading OPEN data from: {open_csv_path}")
    open_data = np.loadtxt(open_csv_path, delimiter=',')
    if open_data.ndim == 1:
        open_data = open_data.reshape(1, -1)

    print(f"OPEN data shape (from file): {open_data.shape}")

    print(f"Loading CLOSED data from: {closed_csv_path}")
    closed_data = np.loadtxt(closed_csv_path, delimiter=',')
    if closed_data.ndim == 1:
        closed_data = closed_data.reshape(1, -1)

    print(f"CLOSED data shape (from file): {closed_data.shape}")

    # After loading open_data and closed_data:
    open_data, closed_data, keep_mask = remove_constant_channels(open_data, closed_data)


    sampling_rate = BoardShim.get_sampling_rate(board_id)

    print("\nExtracting features for OPEN eyes...")
    X_open, y_open = extract_features_from_recording(
        open_data, sampling_rate, window_sec, label=1
    )
    print("OPEN: X shape:", X_open.shape)

    print("Extracting features for CLOSED eyes...")
    X_closed, y_closed = extract_features_from_recording(
        closed_data, sampling_rate, window_sec, label=0
    )
    print("CLOSED: X shape:", X_closed.shape)

    # Combine datasets
    X = np.vstack([X_open, X_closed])
    y = np.concatenate([y_open, y_closed])

    print("\nFull dataset shape:", X.shape, "Labels shape:", y.shape)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,        # 20% for testing
        random_state=42,
        stratify=y            # preserve class balance
    )

    print("\nTrain set shape:", X_train.shape, "Test set shape:", X_test.shape)

    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate on TRAIN set
    y_train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_cm = confusion_matrix(y_train, y_train_pred)
    train_report = classification_report(
        y_train,
        y_train_pred,
        target_names=["CLOSED (0)", "OPEN (1)"]
    )

    print("\n============ MODEL EVALUATION (TRAIN SET) ============")
    print(f"Train Accuracy: {train_acc:.3f}")
    print("\nTrain Confusion matrix (rows = true, cols = predicted):")
    print(train_cm)
    print("\nTrain Classification report:")
    print(train_report)
    print("======================================================")

    # Evaluate on TEST set
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    test_report = classification_report(
        y_test,
        y_test_pred,
        target_names=["CLOSED (0)", "OPEN (1)"]
    )

    print("\n============ MODEL EVALUATION (TEST SET) =============")
    print(f"Test Accuracy: {test_acc:.3f}")
    print("\nTest Confusion matrix (rows = true, cols = predicted):")
    print(test_cm)
    print("\nTest Classification report:")
    print(test_report)
    print("======================================================\n")

    # Save model + metadata
    model_package = {
        "model": clf,
        "sampling_rate": sampling_rate,
        "board_id": board_id,
        "n_channels": open_data.shape[0],
        "window_sec": window_sec,
    }
    joblib.dump(model_package, model_path)
    print(f"Model trained and saved to: {model_path}")

    # Return stats programmatically
    return {
        "model": clf,
        "train_accuracy": train_acc,
        "train_confusion_matrix": train_cm,
        "train_classification_report": train_report,
        "test_accuracy": test_acc,
        "test_confusion_matrix": test_cm,
        "test_classification_report": test_report,
    }


# ==========================
# 3) Live prediction
# ==========================

def live_eye_state_prediction(
    model_path: str = DEFAULT_MODEL_PATH,
    serial_port: str = SERIAL_PORT,
    board_id: int = DEFAULT_BOARD_ID
):
    """
    Uses a stored model to predict eye state on live data.

    - Loads a model saved by train_and_save_eye_state_model.
    - Continuously collects windows of data from the board.
    - When prediction is CLOSED eyes (label 0), plays a sound.

    Press Ctrl+C to stop.
    """
    # Load model package
    model_package = joblib.load(model_path)
    clf = model_package["model"]
    trained_sampling_rate = model_package["sampling_rate"]
    trained_board_id = model_package["board_id"]
    window_sec = model_package["window_sec"]

    if trained_board_id != board_id:
        print(
            "WARNING: Model was trained on board_id "
            f"{trained_board_id}, but you're using board_id {board_id} now."
        )

    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = serial_port

    board = BoardShim(board_id, params)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    window_size = int(window_sec * sampling_rate)

    print("\nStarting live prediction:")
    print(f"  Serial port    : {serial_port}")
    print(f"  Board ID       : {board_id}")
    print(f"  Sampling rate  : {sampling_rate} Hz "
          f"(trained at {trained_sampling_rate} Hz)")
    print(f"  Window size    : {window_sec} s -> {window_size} samples")

    board.prepare_session()
    board.start_stream()
    try:
        print("\nLive prediction running. Press Ctrl+C to stop.\n")
        while True:
            time.sleep(window_sec)

            # Get latest data
            data = board.get_current_board_data(window_size)
            eeg_data = data[eeg_channels, :]

            if eeg_data.shape[1] < window_size:
                # Not enough data yet
                continue

            # Use last window
            window = eeg_data[:, -window_size:]
            feats = extract_features_from_window(window).reshape(1, -1)
            pred = clf.predict(feats)[0]

            label_str = "OPEN" if pred == 1 else "CLOSED"
            print(f"Prediction: {label_str}")

            if pred == 0:  # CLOSED eyes
                play_beep()

    except KeyboardInterrupt:
        print("\nStopping live prediction (KeyboardInterrupt).")
    finally:
        board.stop_stream()
        board.release_session()
        print("Session closed.")


# ==========================
# 4) Predict from CSV file
# ==========================

def predict_eye_state_from_csv(
    csv_path: str,
    model_path: str = DEFAULT_MODEL_PATH
):
    """
    Load a CSV file (same format as training CSVs: (n_channels, n_samples))
    and predict eye state for each window in the recording.

    Returns:
        preds: np.ndarray of shape (n_windows,)
               (0 = CLOSED, 1 = OPEN)
    """
    print(f"\nLoading data from CSV for prediction: {csv_path}")
    eeg_data = np.loadtxt(csv_path, delimiter=',')
    if eeg_data.ndim == 1:
        eeg_data = eeg_data.reshape(1, -1)
    print("CSV data shape:", eeg_data.shape)

    model_package = joblib.load(model_path)
    clf = model_package["model"]
    sampling_rate = model_package["sampling_rate"]
    window_sec = model_package["window_sec"]

    window_size = int(window_sec * sampling_rate)
    n_channels, n_samples = eeg_data.shape
    n_windows = n_samples // window_size

    if n_windows == 0:
        raise ValueError(
            f"Not enough samples ({n_samples}) for one window of size {window_size}"
        )

    print(f"Using window size: {window_size} samples "
          f"({window_sec} s at {sampling_rate} Hz)")
    print(f"Number of windows: {n_windows}")

    X_list = []
    for w in range(n_windows):
        start = w * window_size
        end = start + window_size
        window = eeg_data[:, start:end]
        feats = extract_features_from_window(window)
        X_list.append(feats)

    X = np.vstack(X_list)
    preds = clf.predict(X)

    # Simple summary
    n_closed = np.sum(preds == 0)
    n_open = np.sum(preds == 1)

    print("\n========== PREDICTION FROM CSV ==========")
    print(f"Total windows: {len(preds)}")
    print(f"CLOSED (0): {n_closed}")
    print(f"OPEN   (1): {n_open}")
    print("=========================================")

    return preds


# ==========================
# Example main: collect & train
# ==========================

def main():
    """
    Example flow:
      1) Prepare board session
      2) Collect open/closed data and save to CSV
      3) Train + evaluate + save model using the most recent CSVs

    For live prediction, run `live_eye_state_prediction()` separately.
    """
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT

    board = BoardShim(DEFAULT_BOARD_ID, params)
    board.prepare_session()

    try:
        # 1) Collect training data (this will also save CSVs)
        if(True):
            _, _, _, _ = collect_eye_state_data(
                        board,
                        board_id=DEFAULT_BOARD_ID,
                        duration_per_state_sec=60,  # 1 minute per state
                        pre_beep_sec=5,
                        save_dir=DATA_DIR
                    )

        # 2) Get most recent OPEN and CLOSED CSV files
        #latest_open_csv = get_most_recent_csv("OPEN_", DATA_DIR)
        #latest_closed_csv = get_most_recent_csv("CLOSED_", DATA_DIR)

        # 2) Combine ALL existing OPEN_*.csv and CLOSED_*.csv files in DATA_DIR
        latest_open_csv = combine_csv_files_with_prefix("OPEN_", DATA_DIR)
        latest_closed_csv = combine_csv_files_with_prefix("CLOSED_", DATA_DIR)

        # 3) Train, evaluate, save model
        train_and_save_eye_state_model(
            latest_open_csv,
            latest_closed_csv,
            board_id=DEFAULT_BOARD_ID,
            window_sec=1,
            model_path=DEFAULT_MODEL_PATH
        )

    finally:
        board.release_session()
        print("Training session closed.")


if __name__ == "__main__":
    #predict_eye_state_from_csv("data/OPEN_20251121_153313.csv")
    #predict_eye_state_from_csv("data/OPEN_20251121_152528.csv")
    #predict_eye_state_from_csv("data/CLOSED_20251121_153422.csv")
    #predict_eye_state_from_csv("data/CLOSED_20251121_152638.csv")

    #exit(0)
    main()
    # For live prediction you can instead / later run:
    #exit(0)
    live_eye_state_prediction(
         model_path=DEFAULT_MODEL_PATH,
         serial_port=SERIAL_PORT,
         board_id=DEFAULT_BOARD_ID
     )

#Filtering and Impedance:
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations


def band_filter(
        eeg_data: np.ndarray,
        sampling_rate: int,
        lowcut: float = 1.0,
        highcut: float = 40.0,
        notch: float = 50.0  # use 60.0 for US
):
    """
    Apply band-pass and optional notch filtering to EEG data.

    eeg_data: shape (n_channels, n_samples)
    sampling_rate: sampling rate of board
    lowcut, highcut: band-pass frequencies in Hz
    notch: notch filter frequency (set None to disable)

    Returns filtered copy of eeg_data.
    """
    filtered = eeg_data.copy()

    for ch in range(filtered.shape[0]):
        # Band-pass filter
        DataFilter.perform_bandpass(
            filtered[ch],
            sampling_rate,
            (lowcut + highcut) / 2,  # center frequency
            (highcut - lowcut) / 2,  # bandwidth
            4,
            FilterTypes.BUTTERWORTH.value,
            0
        )

        # Optional notch filter (remove powerline noise)
        if notch is not None:
            DataFilter.perform_bandstop(
                filtered[ch],
                sampling_rate,
                notch,
                2.0,  # bandwidth
                4,
                FilterTypes.BUTTERWORTH.value,
                0
            )

    return filtered


#NOTES:
"""
The seems that the model works more or less but needs more data because it seems to be very
overfitting. 100% is too much.

TODO:
collect the data by changing OPEN and CLOSED several times like within 10s
put the computer saying open eyes. close eyes.. and starts recording after 2s.
"""