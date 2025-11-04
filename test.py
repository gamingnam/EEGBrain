from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time
import numpy as np

def main():
    BoardShim.enable_dev_board_logger()  # enable debug logging

    # ---- SETUP PARAMETERS ----
    params = BrainFlowInputParams()
    # Change this to your actual serial port!
    # macOS/Linux: usually /dev/ttyUSB0 or /dev/tty.usbserial-Dxxxx
    #params.serial_port = "/dev/ttyUSB0"

    # Create Cyton board object (8-channel)
    board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)

    try:
        print("Preparing session...")
        board.prepare_session()

        print("Starting data stream...")
        board.start_stream()

        # Collect 10 seconds of data
        time.sleep(10)

        print("Stopping stream...")
        data = board.get_board_data()  # get all collected data
        board.stop_stream()
        board.release_session()

        # ---- PRINT RESULTS ----
        eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
        print("EEG channels:", eeg_channels)
        print("Data shape (rows=channels, cols=samples):", data.shape)

        # Show first 10 samples from first EEG channel
        ch0 = data[eeg_channels[0]]
        print("First 10 samples of channel 0:", ch0[:10])

    except Exception as e:
        print("Error:", e)
        board.release_session()

if __name__ == "__main__":
    main()

