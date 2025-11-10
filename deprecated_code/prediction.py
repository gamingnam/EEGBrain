import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# ------------------------------
# 1. Setup BrainFlow (synthetic board for now)
# ------------------------------


BoardShim.enable_dev_board_logger()

params = BrainFlowInputParams()
params.serial_port = "/dev/cu.usbserial-DP04VYIJ"
board_id = BoardIds.CYTON_BOARD.value  # Use CYTON_BOARD.value later
board = BoardShim(board_id, params)


board.prepare_session()
board.start_stream()

sfreq = BoardShim.get_sampling_rate(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)

print("EEG channels:", eeg_channels)
print("Sampling rate:", sfreq)

# ------------------------------
# 2. Collect training data (fake labels for demo)
# ------------------------------
print("Collecting training data...")
time.sleep(3)  # let buffer fill

train_data = board.get_board_data()  # shape: (num_channels, num_samples)
X_train = []
y_train = []

# Create features (bandpower 8–30 Hz for each EEG channel)
for i, ch in enumerate(eeg_channels):
    signal = train_data[ch, :]
    DataFilter.perform_bandpass(signal, sfreq,
                                19.0, 22.0, 4,
                                FilterTypes.BUTTERWORTH.value, 0)
    power = np.mean(signal ** 2)
    X_train.append(power)

# Fake labels for demo (alternate left=0, right=1)
y_train = [0 if i % 2 == 0 else 1 for i in range(len(X_train))]

X_train = np.array(X_train).reshape(-1, 1)
y_train = np.array(y_train)

# ------------------------------
# 3. Train classifier
# ------------------------------
clf = LDA()
clf.fit(X_train, y_train)
print("Trained LDA with fake labels.")

# ------------------------------
# 4. Real-time loop: predict and show light
# ------------------------------
plt.ion()
fig, ax = plt.subplots()
circle = plt.Circle((0.5, 0.5), 0.3, color="grey")
ax.add_patch(circle)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

print("Starting online classification... Press Ctrl+C to stop.")
try:
    while True:
        data = board.get_current_board_data(256)  # grab ~1s of data
        features = []
        for ch in eeg_channels:
            signal = data[ch, :]
            DataFilter.perform_bandpass(signal, sfreq,
                                        19.0, 22.0, 4,
                                        FilterTypes.BUTTERWORTH.value, 0)

            power = np.mean(signal ** 2)
            features.append(power)
        features = np.array(features).reshape(-1, 1)

        # Predict class (0=left→green, 1=right→red)
        pred = clf.predict(features)[0]
        if pred == 0:
            circle.set_color("green")
        else:
            circle.set_color("red")

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopping...")
    board.stop_stream()
    board.release_session()
    plt.close(fig)
