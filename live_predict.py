# live_predict.py
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes
from utils import speak

def run_live_prediction(model_path):
    speak("Loading model for live prediction.")
    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    # Board setup
    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DP04VYIJ"
    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    board.prepare_session()
    board.start_stream()

    sfreq = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)

    # Visualization setup
    plt.ion()
    fig, ax = plt.subplots()
    circle = plt.Circle((0.5, 0.5), 0.3, color="gray")
    ax.add_patch(circle)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    speak("Starting live eye state detection. Press Control + C to stop.")

    try:
        while True:
            # Collect about 1 second of EEG data
            data = board.get_current_board_data(256)
            alpha_powers = []

            # Compute alpha power across all EEG channels
            for ch in eeg_channels:
                signal = data[ch, :]
                DataFilter.perform_bandpass(signal, sfreq, 8.0, 12.0, 4,
                                            FilterTypes.BUTTERWORTH.value, 0)
                power = np.mean(signal ** 2)
                alpha_powers.append(power)

            # Use mean alpha power (or modify to use first channel only)
            avg_power = np.mean(alpha_powers)
            pred = clf.predict([[avg_power]])[0]

            # Visual & audio feedback
            if pred == 0:
                circle.set_color("green")
                state = "Eyes open"
            else:
                circle.set_color("blue")
                state = "Eyes closed"

            fig.canvas.draw()
            fig.canvas.flush_events()

            # Print diagnostics
            print("─" * 50)
            print(f"Prediction: {state.upper()}")
            for i, power in enumerate(alpha_powers):
                print(f"  Channel {eeg_channels[i]} alpha power: {power:.6f}")
            print("─" * 50)

            speak(state)
            time.sleep(1)

    except KeyboardInterrupt:
        board.stop_stream()
        board.release_session()
        plt.close(fig)
        speak("Session ended.")
