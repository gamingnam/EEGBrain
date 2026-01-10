import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations, WaveletTypes


def main():
    BoardShim.enable_dev_board_logger()

    # use synthetic board for demo
    params = BrainFlowInputParams()
    board_id = BoardIds.SYNTHETIC_BOARD.value
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.config_board('impedance_mode:1')
    board.start_stream()
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    for i in range(5):
        time.sleep(1)
        data = board.get_board_data()  # get all data and remove it from internal buffer
        eeg_data = data[eeg_channels, :]
        print(f'{data.shape[0]} channels x {data.shape[1]} samples')

    for count, channel in enumerate(eeg_channels):

    board.stop_stream()

    ##### OpenBCI code to calculate impedance:
    for (int Ichan=0; Ichan < nchan; Ichan++) is_railed[Ichan].update(dataProcessingRawBuffer[Ichan], Ichan);

    // compute
    the
    electrode
    impedance.Do
    it in a
    very
    simple
    way[rms
    to
    amplitude, then
    uVolt
    to
    Volt, then
    Volt / Amp
    to
    Ohm]
    for (int Ichan=0; Ichan < nchan; Ichan++) {
                                              // Calculate the impedance
    float impedance = (sqrt(2.0) * dataProcessing.data_std_uV[Ichan] * 1.0e-6) / BoardCytonConstants.leadOffDrive_amps;
    // Subtract the 2.2kOhm resistor
    impedance -= BoardCytonConstants.series_resistor_ohms;
    // Verify the impedance is not less than 0
    if (impedance < 0) {
    // Incase impedance some how dipped below 2.2kOhm
    impedance = 0;
    }
    // Store to the global variable
    data_elec_imp_ohm[Ichan] = impedance;
    }

}


if __name__ == "__main__":
    main()