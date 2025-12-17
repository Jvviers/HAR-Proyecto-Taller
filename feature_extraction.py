import pandas as pd
import numpy as np
from scipy import stats
import os
import glob

# MHEALTH Dataset Constants
SAMPLING_RATE = 50  # Hz
WINDOW_SECONDS = 2  # seconds
WINDOW_SAMPLES = int(SAMPLING_RATE * WINDOW_SECONDS)  # 100 samples
OVERLAP_PERCENT = 0.5
OVERLAP_SAMPLES = int(WINDOW_SAMPLES * OVERLAP_PERCENT) # 50 samples
NUM_CHANNELS = 23

# Column names (Approximate based on MHEALTH structure)
# 0-2: Acc Chest
# 3-4: ECG
# 5-7: Acc Ankle
# 8-10: Gyro Ankle
# 11-13: Mag Ankle
# 14-16: Acc Arm
# 17-19: Gyro Arm
# 20-22: Mag Arm
# 23: Label

def load_data(data_dir):
    """
    Loads all mhealth_subject*.log files from the data directory.
    Returns a dictionary {subject_id: dataframe}.
    """
    files = glob.glob(os.path.join(data_dir, "mhealth_subject*.log"))
    data = {}
    if not files:
        print(f"Warning: No .log files found in {data_dir}")
        return data

    for f in files:
        # Extract subject ID from filename (assuming mhealth_subject1.log format)
        try:
            filename = os.path.basename(f)
            # Extract subject ID from filename (handle mhealth/mHealth case)
            lower_name = filename.lower()
            subject_id = int(lower_name.replace("mhealth_subject", "").replace(".log", ""))
            
            # Read file - simplified parsing assuming space/tab separation
            df = pd.read_csv(f, header=None, delim_whitespace=True)
            
            # Name columns
            cols = [f"ch_{i}" for i in range(NUM_CHANNELS)] + ["label"]
            if df.shape[1] == 24:
                df.columns = cols
            else:
                # Handle potential mismatch
                print(f"Warning: File {filename} has {df.shape[1]} columns, expected 24.")
                continue

            data[subject_id] = df
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    return data

def calculate_magnitudes(df):
    """
    Adds 3 magnitude columns for accelerometers.
    Assumptions:
    - Chest Acc: ch_0, ch_1, ch_2
    - Ankle Acc: ch_5, ch_6, ch_7
    - Arm Acc: ch_14, ch_15, ch_16
    """
    # Create copy to avoid SettingWithCopy warnings
    df = df.copy()
    
    # Chest
    df['mag_chest'] = np.sqrt(df['ch_0']**2 + df['ch_1']**2 + df['ch_2']**2)
    
    # Ankle
    df['mag_ankle'] = np.sqrt(df['ch_5']**2 + df['ch_6']**2 + df['ch_7']**2)
    
    # Arm
    df['mag_arm'] = np.sqrt(df['ch_14']**2 + df['ch_15']**2 + df['ch_16']**2)
    
    return df

def calculate_statistics(window_data):
    """
    Calculates 7 statistics for each column.
    Stats: Min, Max, Mean, Std, Median, Skew, Kurtosis.
    """
    features = []
    # Calculate stats for all columns (original channels + magnitudes)
    # window_data shape: (samples, channels)
    
    for col in window_data.columns:
        series = window_data[col]
        features.extend([
            series.min(),
            series.max(),
            series.mean(),
            series.std(),
            series.median(),
            stats.skew(series),
            stats.kurtosis(series)
        ])
    return features

def check_purity(labels, threshold=1.0):
    """
    Checks if the window label is pure (all samples have the same label).
    Or meets a threshold percentage. 
    For MHEALTH, usually we want 100% purity or majority voting.
    User said: "DEFUBUR NETRICA PORCENTAJE DECIDIR SI CUMPLE ES LA ACTIVIDAD"
    Let's use majority vote but require high purity (e.g. 90% or 100%).
    Also 0 label is 'Null' class in MHEALTH, usually ignored.
    """
    # Scipy < 1.9 compatibility
    mode_result = stats.mode(labels)
    mode_label = mode_result[0][0]
    
    # Ignore null class (label 0)
    if mode_label == 0:
        return None, False

    # Check purity
    count = np.sum(labels == mode_label)
    purity = count / len(labels)
    
    if purity >= threshold:
        return mode_label, True
    return None, False

def extract_features_from_subject(df, window_size=WINDOW_SAMPLES, overlap=OVERLAP_SAMPLES):
    """
    Processes one subject's dataframe into X (features) and y (labels).
    """
    # 1. Add magnitudes
    df = calculate_magnitudes(df)
    
    # Drop label column for feature calculation
    feature_cols = [c for c in df.columns if c != 'label']
    
    X = []
    y = []
    
    num_samples = len(df)
    step = window_size - overlap
    
    for i in range(0, num_samples - window_size + 1, step):
        window = df.iloc[i : i + window_size]
        labels = window['label'].values
        
        # Check purity (using 100% purity for typical strict activity recognition, or user defined)
        # User said "sacar pureza de eso ... si no se desechca"
        label, is_valid = check_purity(labels, threshold=0.9) # Using 90% tolerance
        
        if is_valid:
            # Extract features
            window_data = window[feature_cols]
            row_features = calculate_statistics(window_data)
            X.append(row_features)
            y.append(label)
            
    return np.array(X), np.array(y)

def extract_features_for_prediction(df, window_size=WINDOW_SAMPLES, overlap=OVERLAP_SAMPLES):
    """
    Processes a raw dataframe (no labels required) into X (features) and time segments.
    """
    # 1. Add magnitudes
    df = calculate_magnitudes(df)
    
    # Use all columns (except potential label if it exists, but usually raw data won't have it or we ignore it)
    # We expect 26 columns (23 sensors + 3 mags)
    # If input has 23 cols, we add 3 mags -> 26.
    # We need to ensure column order/names match what training expected if we rely on names.
    # calculate_statistics iterates over columns.
    
    # Ensure only feature columns are present
    feature_cols = [c for c in df.columns if c not in ['label']]
    
    X = []
    time_segments = [] # (start_sample, end_sample)
    
    num_samples = len(df)
    step = window_size - overlap
    
    for i in range(0, num_samples - window_size + 1, step):
        window = df.iloc[i : i + window_size]
        
        # Extract features
        window_data = window[feature_cols]
        row_features = calculate_statistics(window_data)
        X.append(row_features)
        time_segments.append((i, i + window_size))
            
    return np.array(X), time_segments

def get_feature_names(df_columns):
    """
    Generate feature names for the columns.
    """
    stats_names = ['min', 'max', 'mean', 'std', 'median', 'skew', 'kurt']
    feature_names = []
    for col in df_columns:
        if col != 'label':
            for stat in stats_names:
                feature_names.append(f"{col}_{stat}")
    return feature_names
