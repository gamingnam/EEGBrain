import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# ----------------------------------------------
# CONFIGURATION
OPENBCI_BASE_DIR = "/Users/namdang/Documents/OpenBCI_GUI/Recordings"
OUTPUT_DIR = "/Users/namdang/Desktop/ML/data"
DEFAULT_SAMPLE_RATE = 255.0  # Hz
# ----------------------------------------------

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def read_openbci_csv(filepath):
    """
    Reads an OpenBCI raw data file (.txt or .csv).
    Skips metadata lines starting with '%' and parses the header correctly.
    """
    try:
        df = pd.read_csv(filepath, comment='%', sep=',', engine='python')
        print(f"  → successfully read file: {os.path.basename(filepath)}")
        print(f"  → columns detected: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"  ✖ Failed to parse file {filepath}: {e}")
        return None

def guess_eeg_columns(df):
    candidates = []
    for col in df.columns:
        name = str(col).lower()
        if any(k in name for k in ("exg", "eeg", "chan", "channel", "ch", "raw", "electrode")):
            candidates.append(col)
    if not candidates:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) > 1:
            candidates = numeric_cols[1:9]
        else:
            candidates = numeric_cols
    candidates = [c for c in candidates if 'accel' not in str(c).lower() and 'gyro' not in str(c).lower()]
    if not candidates:
        candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return list(dict.fromkeys(candidates))

def clean_file(filepath, lowcut=1, highcut=50, fs=DEFAULT_SAMPLE_RATE):
    print(f"\nProcessing: {filepath}")
    print(f"  → sample rate: {fs} Hz")

    df = read_openbci_csv(filepath)
    if df is None or df.empty:
        print("  ✖ Skipping: could not read file.")
        return None

    eeg_cols = guess_eeg_columns(df)
    if not eeg_cols:
        print("  ✖ No EEG columns detected — skipping file.")
        return None

    print(f"  → detected EEG columns: {eeg_cols}")

    for ch in eeg_cols:
        df[ch] = pd.to_numeric(df[ch], errors='coerce')
        if np.isfinite(df[ch]).sum() < 10:
            print(f"    - skip filtering {ch}: not enough numeric data")
            continue
        try:
            df[ch] = bandpass_filter(df[ch].fillna(method='ffill').fillna(method='bfill'), lowcut, highcut, fs)
        except Exception as e:
            print(f"    - warning: filtering failed for {ch}: {e}")

    try:
        df[eeg_cols] = (df[eeg_cols] - df[eeg_cols].mean()) / df[eeg_cols].std()
    except Exception as e:
        print(f"  ! normalization failed: {e}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.basename(filepath)
    # Change extension to .csv if it's a .txt file
    if base.lower().endswith('.txt'):
        base = base[:-4] + '.csv'
    outpath = os.path.join(OUTPUT_DIR, f"cleaned_{base}")

    try:
        df.to_csv(outpath, index=False)
        print(f"  ✅ saved cleaned file to: {outpath}")
    except Exception as e:
        print(f"  ✖ failed to save cleaned file: {e}")
        return None

    return outpath

def main():
    if not os.path.exists(OPENBCI_BASE_DIR):
        print(f"Error: Base directory not found: {OPENBCI_BASE_DIR}")
        return

    # Find all folders starting with "OpenBCISession_"
    session_dirs = [
        os.path.join(OPENBCI_BASE_DIR, d)
        for d in os.listdir(OPENBCI_BASE_DIR)
        if d.startswith("OpenBCISession_") and os.path.isdir(os.path.join(OPENBCI_BASE_DIR, d))
    ]

    if not session_dirs:
        print(f"No session folders found in {OPENBCI_BASE_DIR}")
        return

    print(f"Found {len(session_dirs)} OpenBCI session folder(s):")
    for s in session_dirs:
        print(f"  - {os.path.basename(s)}")

    # Iterate through all .txt and .csv files in those folders
    csv_files = []
    for sdir in session_dirs:
        for file in os.listdir(sdir):
            if file.lower().endswith(('.csv', '.txt')):
                csv_files.append(os.path.join(sdir, file))

    if not csv_files:
        print("No .csv or .txt files found in any session folders.")
        return

    print(f"\nFound {len(csv_files)} total data file(s) to process.")
    for fp in csv_files:
        clean_file(fp)

    # Open Finder to show output folder (macOS only)
    print(f"\n🗂️  Opening output folder: {OUTPUT_DIR}")
    os.system(f"open '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
