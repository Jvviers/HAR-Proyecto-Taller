import argparse
import pandas as pd
import numpy as np
import onnxruntime as rt
import matplotlib.pyplot as plt
import train_model as fe # Changed from feature_extraction to train_model
import os
import joblib

def predict_timeline(log_file, model_path="rf_mhealth.onnx"):
    # 1. Load Data
    print(f"Loading {log_file}...")
    try:
        df = pd.read_csv(log_file, header=None, delim_whitespace=True)
        # Handle columns
        if df.shape[1] >= fe.NUM_CHANNELS:
             # Take first 23 columns
            df = df.iloc[:, :fe.NUM_CHANNELS]
            cols = [f"ch_{i}" for i in range(fe.NUM_CHANNELS)]
            df.columns = cols
        else:
            print(f"Error: File has {df.shape[1]} columns, expected at least {fe.NUM_CHANNELS}.")
            return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 2. Extract Features
    print("Extracting features...")
    X, time_segments = fe.extract_features_for_prediction(df)
    
    if len(X) == 0:
        print("No valid windows found.")
        return

    # 2b. Scale Features
    print("Scaling features...")
    try:
        scaler = joblib.load("scaler.joblib")
        X = scaler.transform(X)
    except Exception as e:
        print(f"Error loading scaler: {e}")
        print("Ensure you have run train_model.py first to generate scaler.joblib")
        return
        
    # 3. Load ONNX Model
    print(f"Loading model from {model_path}...")
    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    
    # 4. Predict
    print("Predicting...")
    # ONNX Runtime expects float32
    X = X.astype(np.float32)
    pred_onx = sess.run([label_name], {input_name: X})[0]
    
    # 5. Visualize Timeline
    print("Generating Timeline...")
    plot_timeline(time_segments, pred_onx, log_file)

def plot_timeline(time_segments, predictions, filename):
    # Create a list of (start_time, duration, activity_code)
    # Sampling rate 50Hz -> Time = sample / 50
    sr = fe.SAMPLING_RATE
    
    segments = []
    for (start, end), pred in zip(time_segments, predictions):
        start_sec = start / sr
        duration_sec = (end - start) / sr
        segments.append((start_sec, duration_sec, pred))
        
    # Consolidate segments for cleaner plotting
    # (Optional optimization, but let's plot raw windows first)
    
    fig, ax = plt.subplots(figsize=(15, 2))
    
    # Map activities to colors
    # Activities 0-12 usually.
    unique_activities = np.unique(predictions)
    cmap = plt.get_cmap('tab20')
    norm = plt.Normalize(vmin=0, vmax=12) # Assuming max 12 activities
    
    for start, duration, act in segments:
        color = cmap(norm(act))
        ax.broken_barh([(start, duration)], (10, 10), facecolors=color, edgecolor='none')
        
    ax.set_ylim(5, 25)
    ax.set_xlim(0, segments[-1][0] + segments[-1][1])
    ax.set_xlabel('Time (seconds)')
    ax.set_yticks([])
    ax.set_title(f'Activity Timeline: {os.path.basename(filename)}')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cmap(norm(act)), label=f'Activity {act}') 
                       for act in unique_activities]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=5)
    
    output_img = filename.replace(".log", "_timeline.png")
    plt.tight_layout()
    plt.savefig(output_img)
    plt.show()
    print(f"Timeline saved to {output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict activity from MHEALTH log.")
    parser.add_argument("logfile", help="Path to the .log file")
    parser.add_argument("--model", default="rf_mhealth.onnx", help="Path to ONNX model")
    
    args = parser.parse_args()
    predict_timeline(args.logfile, args.model)
