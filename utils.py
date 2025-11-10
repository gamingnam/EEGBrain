# utils.py
import os
import time
import pyttsx3
import pandas as pd

def speak(text):
    print(f"[TTS] {text}")
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S")

def choose_model():
    model_dir = "models/trained_models"
    os.makedirs(model_dir, exist_ok=True)
    models = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    if not models:
        speak("No saved models found. Training a new one.")
        return None

    print("\nAvailable models:")
    for i, m in enumerate(models):
        print(f"{i + 1}: {m}")

    choice = input("Enter model number or 'n' for new: ").strip().lower()
    if choice == "n":
        return None
    try:
        idx = int(choice) - 1
        return os.path.join(model_dir, models[idx])
    except:
        speak("Invalid choice. Training a new model.")
        return None

def save_csv_with_header(data, filepath):
    columns = [
        "Sample Index", "EXG Channel 0", "EXG Channel 1", "EXG Channel 2",
        "EXG Channel 3", "EXG Channel 4", "EXG Channel 5", "EXG Channel 6",
        "EXG Channel 7", "Accel Channel 0", "Accel Channel 1", "Accel Channel 2",
        "Not Used", "Digital Channel 0 (D11)", "Digital Channel 1 (D12)",
        "Digital Channel 2 (D13)", "Digital Channel 3 (D17)", "Not Used.1",
        "Digital Channel 4 (D18)", "Analog Channel 0", "Analog Channel 1",
        "Analog Channel 2", "Timestamp", "Marker Channel", "Timestamp (Formatted)"
    ]
    df = pd.DataFrame(data.T)
    df.columns = columns[:df.shape[1]]
    df.to_csv(filepath, index=False)
