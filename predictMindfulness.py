import argparse
import time
import serial

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams


def main():
    BoardShim.enable_board_logger()
    DataFilter.enable_data_logger()
    MLModel.enable_ml_logger()

    arduino = serial.Serial('/dev/cu.usbserial-110', baudrate=115200, timeout=.1)

    params = BrainFlowInputParams()
    #params.serial_port = args.serial_port
    SERIAL_PORT = '/dev/cu.usbserial-DP04VYIJ'
    params.serial_port = SERIAL_PORT
    DEFAULT_BOARD_ID = BoardIds.CYTON_BOARD.value

    board = BoardShim(DEFAULT_BOARD_ID, params)
    master_board_id = board.get_board_id()
    sampling_rate = BoardShim.get_sampling_rate(master_board_id)
    board.prepare_session()
    board.start_stream(45000)
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    time.sleep(5)  # recommended window size for eeg metric calculation is at least 4 seconds, bigger is better
    data = board.get_board_data()
    eeg_channels = BoardShim.get_eeg_channels(master_board_id)
    board.stop_stream()
    board.release_session()

    bands = DataFilter.get_avg_band_powers(data, eeg_channels, sampling_rate, True)
    feature_vector = bands[0]
    print(feature_vector)

    mindfulness_params = BrainFlowModelParams(BrainFlowMetrics.MINDFULNESS.value,
                                              BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
    mindfulness = MLModel(mindfulness_params)
    mindfulness.prepare()
    print('Mindfulness: %s' % str(mindfulness.predict(feature_vector)))
    mindfulness.release()

    restfulness_params = BrainFlowModelParams(BrainFlowMetrics.RESTFULNESS.value,
                                              BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
    restfulness = MLModel(restfulness_params)
    restfulness.prepare()
    print('Restfulness: %s' % str(restfulness.predict(feature_vector)))
    restfulness.release()


if __name__ == "__main__":
    main()




