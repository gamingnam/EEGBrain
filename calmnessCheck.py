import time
import serial
import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams


def main():
    BoardShim.enable_board_logger()
    DataFilter.enable_data_logger()
    MLModel.enable_ml_logger()

    #arduino = serial.Serial('/dev/cu.usbserial-110', baudrate=115200, timeout=.1)



    params = BrainFlowInputParams()
    params.serial_port = '/dev/cu.usbserial-DP04VYIJ'
    board_id = BoardIds.CYTON_BOARD.value

    board = BoardShim(board_id, params)
    sampling_rate = BoardShim.get_sampling_rate(board_id)

    # ---- Prepare board ----
    board.prepare_session()
    board.start_stream(45000)
    print("Stream started. Predicting continuously...")

    # ---- Prepare ML models ONCE ----
    mindfulness_params = BrainFlowModelParams(
        BrainFlowMetrics.MINDFULNESS.value,
        BrainFlowClassifiers.DEFAULT_CLASSIFIER.value
    )
    mindfulness = MLModel(mindfulness_params)
    mindfulness.prepare()

    restfulness_params = BrainFlowModelParams(
        BrainFlowMetrics.RESTFULNESS.value,
        BrainFlowClassifiers.DEFAULT_CLASSIFIER.value
    )
    restfulness = MLModel(restfulness_params)
    restfulness.prepare()

    eeg_channels = BoardShim.get_eeg_channels(board_id)

    window_size_sec = 4
    window_samples = window_size_sec * sampling_rate


    try:
        while True:
            # ---- Get last 4 seconds of data ----
            data = board.get_current_board_data(window_samples)

            # Need enough samples to compute band powers
            if data.shape[1] < window_samples:
                time.sleep(0.1)
                continue

            # ---- Compute band powers ----
            bands = DataFilter.get_avg_band_powers(
                data, eeg_channels, sampling_rate, True
            )
            feature_vector = bands[0]

            # ---- Predict ----
            m = mindfulness.predict(feature_vector)[0]
            r = restfulness.predict(feature_vector)[0]

            print(f"Mindfulness: {m:.3f}   Restfulness: {r:.3f}")

            now = time.time()

    except KeyboardInterrupt:
        print("Stopping...")

    # Cleanup
    board.stop_stream()
    board.release_session()
    mindfulness.release()
    restfulness.release()


if __name__ == "__main__":
    main()
