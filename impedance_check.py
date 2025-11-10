# impedance_check.py
import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from utils import speak

def check_impedance():
    speak("Starting impedance check. Please remain still.")

    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DP04VYIJ"
    board = BoardShim(BoardIds.CYTON_BOARD.value, params)

    # Prepare and connect
    board.prepare_session()
    board.config_board("z")  # 'z' → enter impedance test mode
    board.start_stream()

    time.sleep(5)  # let data stream stabilize
    data = board.get_board_data()

    board.config_board("Z")  # 'Z' → exit impedance test mode
    board.stop_stream()

    eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
    sfreq = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)

    speak("Analyzing electrode contact quality.")
    print("\n=== Impedance Check Results ===")
    quality_results = []

    for ch in eeg_channels:
        signal = data[ch, :]
        rms = np.sqrt(np.mean(np.square(signal)))

        # heuristic thresholds (adjust as needed)
        if rms < 10:
            quality = "⚠️ Poor contact"
        elif rms < 50:
            quality = "✅ Acceptable"
        else:
            quality = "🌟 Excellent"

        quality_results.append((ch, quality, round(rms, 2)))
        print(f"Channel {ch}: {quality} (RMS = {rms:.2f})")

    # Give spoken feedback summary
    bad_channels = [ch for ch, q, _ in quality_results if "Poor" in q]
    if bad_channels:
        speak(f"Warning. Poor contact detected on channels {', '.join(map(str, bad_channels))}. Please adjust electrodes.")
    else:
        speak("All channels have acceptable contact.")

    board.release_session()
    print("\nImpedance test complete. You may proceed to training.")
