# main.py
from impedance_check import check_impedance
from train_model import train_new_model
from live_predict import run_live_prediction
from utils import speak, choose_model

def main():
    model_path = choose_model()  # Let user select existing or train new

    check_impedance()  # Verify electrodes before continuing
    if model_path is None:
        model_path = train_new_model()

    run_live_prediction(model_path)

if __name__ == "__main__":
    main()
