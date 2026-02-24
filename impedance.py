import time
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes


def get_cyton_impedances_8ch(board, settle_time=3.0):
    """
    Measure electrode impedance for OpenBCI Cyton (8 EEG channels).

    Returns:
        dict: { 'CH1': impedance_ohms, ..., 'CH8': impedance_ohms }
    """

    board_id = BoardIds.CYTON_BOARD.value
    fs = BoardShim.get_sampling_rate(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)

    window_size = int(settle_time * fs)

    # ------------------------------
    # Enable impedance mode (Cyton)
    # ------------------------------
    for ch in range(1, 9):
        board.config_board(f"z{ch}")
        time.sleep(0.05)

    # ------------------------------
    # Stream data
    # ------------------------------
    board.start_stream(45000)
    time.sleep(settle_time)

    # IMPORTANT: use rolling buffer (same as your working code)
    data = board.get_current_board_data(window_size)

    board.stop_stream()

    impedances = {}

    # ------------------------------
    # Process each EEG channel
    # ------------------------------
    for i, ch in enumerate(eeg_channels):
        signal = data[ch].copy()

        # Bandpass around Cyton impedance tone (~31.25 Hz)
        DataFilter.perform_bandpass(
            signal,
            fs,
            29.0,     # low cut (Hz)
            33.0,     # high cut (Hz)
            4,
            FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
            0
        )

        # RMS voltage in microvolts
        rms_uv = np.sqrt(np.mean(signal ** 2))

        # OpenBCI impedance formula
        impedance_ohms = (np.sqrt(2) * rms_uv * 1e-6 / 6e-9) - 2200
        impedance_ohms = max(0, impedance_ohms)

        impedances[f"CH{i+1}"] = int(impedance_ohms)

    # ------------------------------
    # Disable impedance mode
    # ------------------------------
    for ch in range(1, 9):
        board.config_board(f"z{ch}0")
        time.sleep(0.05)

    return impedances


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DP04VYIJ"  # <-- CHANGE THIS

    board = BoardShim(BoardIds.CYTON_BOARD.value, params)

    try:
        board.prepare_session()

        impedances = get_cyton_impedances_8ch(board)

        print("\nCyton Electrode Impedances:")
        for ch, z in impedances.items():
            print(f"{ch}: {z} Ω")

    finally:
        if board.is_prepared():
            board.release_session()
