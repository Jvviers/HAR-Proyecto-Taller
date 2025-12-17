import os
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN ---
DATA_DIR = "MHEALTHDATASET"
PROCESSED_DIR = "output"
WINDOW_SIZE = 128
OVERLAP = 64
SUBJECT_ID = 10

# Estructura de columnas (Mismas que en training)
FEATURE_COLS = [
    'acc_chest_x', 'acc_chest_y', 'acc_chest_z',
    'ecg_1', 'ecg_2',
    'acc_ankle_x', 'acc_ankle_y', 'acc_ankle_z',
    'gyro_ankle_x', 'gyro_ankle_y', 'gyro_ankle_z',
    'mag_ankle_x', 'mag_ankle_y', 'mag_ankle_z',
    'acc_arm_x', 'acc_arm_y', 'acc_arm_z',
    'gyro_arm_x', 'gyro_arm_y', 'gyro_arm_z',
    'mag_arm_x', 'mag_arm_y', 'mag_arm_z'
]

ACTIVITY_MAP = {
    1: "Standing still (De pie)",
    2: "Sitting and relaxing (Sentado)",
    3: "Lying down (Acostado)",
    4: "Walking (Caminando)",
    5: "Climbing stairs (Subiendo escaleras)",
    6: "Waist bends forward (Flexiones cint.)",
    7: "Frontal elevation of arms (Elev. brazos)",
    8: "Knees bending (Sentadillas)",
    9: "Cycling (Ciclismo)",
    10: "Jogging (Trotar)",
    11: "Running (Correr)",
    12: "Jump front & back (Saltos)"
}

# --- FUNCIONES DE PREPROCESAMIENTO (REUTILIZADAS) ---
def load_subject_data(subject_id):
    """Carga y limpia datos de un sujeto específico."""
    file_name = f"mHealth_subject{subject_id}.log"
    file_path = os.path.join(DATA_DIR, file_name)
    
    # Fallback path
    if not os.path.exists(file_path):
        file_path = os.path.join(DATA_DIR, "MHEALTHDATASET", file_name)

    if not os.path.exists(file_path):
        print(f"Error: Archivo para sujeto {subject_id} no encontrado en {file_path}.")
        return None

    try:
        df = pd.read_csv(file_path, header=None, sep=r'\t', engine='python')
    except Exception as e:
        print(f"Error leyendo archivo {file_path}: {e}")
        return None

    # Asignar columnas
    df.columns = FEATURE_COLS + ['label']
    
    # Eliminar clase Null (0)
    df = df[df['label'] != 0].copy()
    
    # Magnitudes vectoriales
    df['mag_acc_ankle'] = np.sqrt(df['acc_ankle_x']**2 + df['acc_ankle_y']**2 + df['acc_ankle_z']**2)
    df['mag_gyro_ankle'] = np.sqrt(df['gyro_ankle_x']**2 + df['gyro_ankle_y']**2 + df['gyro_ankle_z']**2)
    df['mag_acc_arm'] = np.sqrt(df['acc_arm_x']**2 + df['acc_arm_y']**2 + df['acc_arm_z']**2)
    df['mag_gyro_arm'] = np.sqrt(df['gyro_arm_x']**2 + df['gyro_arm_y']**2 + df['gyro_arm_z']**2)
    
    return df

def calculate_energy(x):
    fft_coeffs = np.fft.rfft(x - np.mean(x))
    return np.sum(np.abs(fft_coeffs)**2) / len(x)

def extract_features_from_window(window_data):
    mean = np.mean(window_data, axis=0)
    std = np.std(window_data, axis=0)
    mins = np.min(window_data, axis=0)
    maxs = np.max(window_data, axis=0)
    medians = np.median(window_data, axis=0)
    skewness = stats.skew(window_data, axis=0)
    energies = np.apply_along_axis(calculate_energy, 0, window_data)
    
    features = np.concatenate([mean, std, mins, maxs, medians, skewness, energies])
    return features

def create_windows(df, window_size=100, overlap=50):
    data_cols = [c for c in df.columns if c != 'label']
    X_raw = df[data_cols].values
    y_raw = df['label'].values
    
    X_windows = []
    y_windows = []
    
    step = window_size - overlap
    num_samples = X_raw.shape[0]
    
    print(f"Generando ventanas para Sujeto {SUBJECT_ID}... (Samples: {num_samples})")
    
    for start in range(0, num_samples - window_size + 1, step):
        end = start + window_size
        window_segment = X_raw[start:end, :]
        
        mode_res = stats.mode(y_raw[start:end])
        try:
            label_mode = mode_res.mode[0]
        except (IndexError, TypeError):
             label_mode = mode_res[0][0] if isinstance(mode_res, tuple) else mode_res.mode # Fallback
        
        features = extract_features_from_window(window_segment)
        X_windows.append(features)
        y_windows.append(label_mode)
        
    return np.array(X_windows), np.array(y_windows)

# --- PIPELINE DE PRUEBA ---
def main():
    print(f"--- TEST CON SUJETO {SUBJECT_ID} ---")
    
    # 1. Cargar datos
    df = load_subject_data(SUBJECT_ID)
    if df is None:
        return

    # 2. Ventaneo y Features
    X_test, y_test = create_windows(df, WINDOW_SIZE, OVERLAP)
    print(f"Ventanas generadas: {X_test.shape}")

    # 3. Cargar Scaler Params y Escalar
    scaler_path = os.path.join(PROCESSED_DIR, 'scaler_params.json')
    if not os.path.exists(scaler_path):
        print("Error: No se encontró scaler_params.json")
        return
        
    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)
        
    mean = np.array(scaler_params["mean"])
    scale = np.array(scaler_params["scale"])
    
    # Aplicar estandarización manualmente: (X - mean) / scale
    X_test_scaled = (X_test - mean) / scale
    
    # Forzar tipo float32 para ONNX
    X_test_scaled = X_test_scaled.astype(np.float32)

    # 4. Cargar Modelo ONNX e Inferencia
    model_path = os.path.join(PROCESSED_DIR, 'har_rf_model.onnx')
    if not os.path.exists(model_path):
        print("Error: No se encontró har_rf_model.onnx")
        return

    print("Cargando modelo ONNX...")
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    
    print("Ejecutando predicciones...")
    y_pred = sess.run([label_name], {input_name: X_test_scaled})[0]

    # 5. Evaluación
    print("\n---------- RESULTADOS SUJETO 10 ----------")
    
    # Mapeo de nombres
    unique_labels = sorted(list(set(y_test) | set(y_pred)))
    target_names = [ACTIVITY_MAP.get(i, f"Unknown-{i}") for i in unique_labels]
    
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))

    # --- CÁLCULO DE TIEMPOS ---
    # Frecuencia muestreo: 50Hz.
    # Step de ventana: WINDOW_SIZE(128) - OVERLAP(64) = 64 muestras.
    # Tiempo por paso: 64 / 50 = 1.28 segundos (aprox, cada ventana "nueva" aporta 1.28s)
    step_duration_sec = (WINDOW_SIZE - OVERLAP) / 50.0
    
    print("\n---------- DURACIÓN ESTIMADA POR ACTIVIDAD (Segundos) ----------")
    print(f"{'Actividad':<45} | {'Real (s)':<10} | {'Predicho (s)':<10}")
    print("-" * 75)
    
    for label_id in unique_labels:
        real_count = np.sum(y_test == label_id)
        pred_count = np.sum(y_pred == label_id)
        
        real_time = real_count * step_duration_sec
        pred_time = pred_count * step_duration_sec
        
        act_name = ACTIVITY_MAP.get(label_id, f"Unknown-{label_id}")
        print(f"{act_name:<45} | {real_time:<10.2f} | {pred_time:<10.2f}")
    print("-" * 75)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusión - Sujeto {SUBJECT_ID}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    output_img = os.path.join(PROCESSED_DIR, f'confusion_matrix_subject{SUBJECT_ID}.png')
    plt.savefig(output_img)
    print(f"Matriz de confusión guardada en: {output_img}")

if __name__ == "__main__":
    main()
