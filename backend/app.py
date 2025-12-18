"""
Backend API para Human Activity Recognition (HAR)
Basado en test_subject10.py - usando modelo ONNX entrenado
"""

import os
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
from scipy import stats
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Habilitar CORS para peticiones del frontend

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR = os.path.join(BASE_DIR, "MHEALTHDATASET")
WINDOW_SIZE = 128
OVERLAP = 64
SAMPLING_RATE = 50  # Hz

# Estructura de columnas
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

# Cargar modelo y scaler al iniciar
model_session = None
scaler_mean = None
scaler_scale = None

def load_model():
    global model_session, scaler_mean, scaler_scale
    
    model_path = os.path.join(PROCESSED_DIR, 'har_rf_model.onnx')
    scaler_path = os.path.join(PROCESSED_DIR, 'scaler_params.json')
    
    if os.path.exists(model_path):
        model_session = ort.InferenceSession(model_path)
        print(f"✓ Modelo ONNX cargado: {model_path}")
    else:
        print(f"✗ Modelo no encontrado: {model_path}")
    
    if os.path.exists(scaler_path):
        with open(scaler_path, 'r') as f:
            params = json.load(f)
        scaler_mean = np.array(params["mean"])
        scaler_scale = np.array(params["scale"])
        print(f"✓ Scaler cargado: {scaler_path}")
    else:
        print(f"✗ Scaler no encontrado: {scaler_path}")

# --- FUNCIONES DE PROCESAMIENTO ---
def calculate_energy(x):
    """Calcula la energía FFT de una señal."""
    fft_coeffs = np.fft.rfft(x - np.mean(x))
    return np.sum(np.abs(fft_coeffs)**2) / len(x)

def extract_features_from_window(window_data):
    """Extrae características estadísticas de una ventana de datos."""
    mean = np.mean(window_data, axis=0)
    std = np.std(window_data, axis=0)
    mins = np.min(window_data, axis=0)
    maxs = np.max(window_data, axis=0)
    medians = np.median(window_data, axis=0)
    skewness = stats.skew(window_data, axis=0)
    energies = np.apply_along_axis(calculate_energy, 0, window_data)
    
    features = np.concatenate([mean, std, mins, maxs, medians, skewness, energies])
    return features

def load_subject_data(subject_id):
    """Carga datos de un sujeto específico."""
    file_name = f"mHealth_subject{subject_id}.log"
    file_path = os.path.join(DATA_DIR, file_name)
    
    # Fallback path si existe subcarpeta
    if not os.path.exists(file_path):
        file_path = os.path.join(DATA_DIR, "MHEALTHDATASET", file_name)
    
    if not os.path.exists(file_path):
        print(f"Archivo no encontrado: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path, header=None, sep=r'\t', engine='python')
        # Verificar columnas antes de asignar
        if df.shape[1] < 24: # Al menos 23 features + label
             print(f"Error: Archivo tiene {df.shape[1]} columnas, se esperaban > 23")
             return None

        # Si hay más columnas, cortamos (a veces el log tiene más cosas)
        if df.shape[1] > 24:
             df = df.iloc[:, :24]

        df.columns = FEATURE_COLS + ['label']
        df = df[df['label'] != 0].copy()
        
        # Magnitudes vectoriales
        df['mag_acc_ankle'] = np.sqrt(df['acc_ankle_x']**2 + df['acc_ankle_y']**2 + df['acc_ankle_z']**2)
        df['mag_gyro_ankle'] = np.sqrt(df['gyro_ankle_x']**2 + df['gyro_ankle_y']**2 + df['gyro_ankle_z']**2)
        df['mag_acc_arm'] = np.sqrt(df['acc_arm_x']**2 + df['acc_arm_y']**2 + df['acc_arm_z']**2)
        df['mag_gyro_arm'] = np.sqrt(df['gyro_arm_x']**2 + df['gyro_arm_y']**2 + df['gyro_arm_z']**2)
        
        return df
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return None

def create_windows(df, window_size=128, overlap=64):
    """Crea ventanas deslizantes con características."""
    data_cols = [c for c in df.columns if c != 'label']
    X_raw = df[data_cols].values
    y_raw = df['label'].values
    
    X_windows = []
    y_windows = []
    
    step = window_size - overlap
    num_samples = X_raw.shape[0]
    
    for start in range(0, num_samples - window_size + 1, step):
        end = start + window_size
        window_segment = X_raw[start:end, :]
        
        # Modo robusto
        try:
            mode_res = stats.mode(y_raw[start:end])
            # Scipy moderno retorna arrays
            if hasattr(mode_res, 'mode'):
                 label_mode = mode_res.mode[0] if np.ndim(mode_res.mode) > 0 else mode_res.mode
            else:
                 # Tuple fallback
                 label_mode = mode_res[0][0]
        except Exception:
             # Fallback absoluto
             vals, counts = np.unique(y_raw[start:end], return_counts=True)
             label_mode = vals[np.argmax(counts)]

        features = extract_features_from_window(window_segment)
        X_windows.append(features)
        y_windows.append(label_mode)
    
    return np.array(X_windows), np.array(y_windows)

def predict_activities(X):
    """Realiza predicciones usando el modelo ONNX."""
    if model_session is None:
        return None
    
    # Escalar datos
    X_scaled = (X - scaler_mean) / scaler_scale
    X_scaled = X_scaled.astype(np.float32)
    
    # Predicción
    input_name = model_session.get_inputs()[0].name
    label_name = model_session.get_outputs()[0].name
    y_pred = model_session.run([label_name], {input_name: X_scaled})[0]
    
    return y_pred

# --- RESULT GENERATORS ---
def generate_prediction_stats(y_test, y_pred, step_duration):
    unique_labels = sorted(list(set(y_test) | set(y_pred)))
    results = []
    
    for label_id in unique_labels:
        real_count = int(np.sum(y_test == label_id))
        pred_count = int(np.sum(y_pred == label_id))
        
        results.append({
            "activity_id": int(label_id),
            "activity_name": ACTIVITY_MAP.get(label_id, f"Unknown-{label_id}"),
            "real_windows": real_count,
            "predicted_windows": pred_count,
            "real_duration_sec": round(real_count * step_duration, 2),
            "predicted_duration_sec": round(pred_count * step_duration, 2)
        })
    
    accuracy = float(np.mean(y_pred == y_test))
    return {
        "total_windows": len(y_test),
        "accuracy": round(accuracy * 100, 2),
        "step_duration_sec": step_duration,
        "activities": results,
        "predictions": y_pred.tolist(),
        "real_labels": y_test.tolist()
    }

def generate_timeline_data(y_test, y_pred, step_duration):
    timeline = []
    for i, (real, pred) in enumerate(zip(y_test, y_pred)):
        timeline.append({
            "window": i,
            "start_time": round(i * step_duration, 2),
            "end_time": round((i + 1) * step_duration, 2),
            "real_activity": int(real),
            "real_name": ACTIVITY_MAP.get(int(real), f"Unknown"),
            "predicted_activity": int(pred),
            "predicted_name": ACTIVITY_MAP.get(int(pred), f"Unknown"),
            "correct": int(real) == int(pred)
        })
    
    return {
        "timeline": timeline,
        "total_duration_sec": round(len(y_test) * step_duration, 2)
    }

def generate_confusion_matrix_data(y_test, y_pred):
    from sklearn.metrics import confusion_matrix
    unique_labels = sorted(list(set(y_test) | set(y_pred)))
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    labels = [ACTIVITY_MAP.get(l, f"Unknown-{l}") for l in unique_labels]
    
    return {
        "labels": labels,
        "label_ids": [int(l) for l in unique_labels],
        "matrix": cm.tolist()
    }



# --- ENDPOINTS API ---
@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint de salud del servidor."""
    return jsonify({
        "status": "ok",
        "model_loaded": model_session is not None,
        "scaler_loaded": scaler_mean is not None
    })

@app.route('/api/activities', methods=['GET'])
def get_activities():
    """Retorna el mapeo de actividades disponibles."""
    return jsonify(ACTIVITY_MAP)

@app.route('/api/subjects', methods=['GET'])
def get_subjects():
    """Retorna los sujetos disponibles."""
    subjects = []
    for i in range(1, 11):
        file_path = os.path.join(DATA_DIR, f"mHealth_subject{i}.log")
        if os.path.exists(file_path):
            subjects.append(i)
    return jsonify({"subjects": subjects})

@app.route('/api/predict/<int:subject_id>', methods=['GET'])
def predict_subject(subject_id):
    """Predice actividades para un sujeto específico."""
    df = load_subject_data(subject_id)
    if df is None:
        return jsonify({"error": f"Sujeto {subject_id} no encontrado"}), 404
    
    X_test, y_test = create_windows(df, WINDOW_SIZE, OVERLAP)
    y_pred = predict_activities(X_test)
    
    if y_pred is None:
        return jsonify({"error": "Modelo no cargado"}), 500
    
    # Calcular duraciones
    step_duration = (WINDOW_SIZE - OVERLAP) / SAMPLING_RATE
    
    stats = generate_prediction_stats(y_test, y_pred, step_duration)
    stats["subject_id"] = subject_id

    return jsonify(stats)

@app.route('/api/predict', methods=['POST'])
def predict_data():
    """Predice actividades a partir de datos enviados en el cuerpo (manual)."""
    data = request.get_json()
    
    if not data or 'sensor_data' not in data:
        return jsonify({"error": "Se requiere 'sensor_data' en el cuerpo"}), 400
    
    try:
        sensor_data = np.array(data['sensor_data'])
        
        if len(sensor_data.shape) == 1:
            # Una sola ventana de características
            X = sensor_data.reshape(1, -1)
        else:
            X = sensor_data
        
        y_pred = predict_activities(X)
        
        if y_pred is None:
            return jsonify({"error": "Modelo no cargado"}), 500
        
        predictions = []
        for pred in y_pred:
            predictions.append({
                "activity_id": int(pred),
                "activity_name": ACTIVITY_MAP.get(int(pred), f"Unknown-{pred}")
            })
        
        return jsonify({
            "predictions": predictions,
            "count": len(predictions)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/predict-file', methods=['POST'])
def predict_file_upload():
    """Recibe un archivo .log/.txt, lo procesa y devuelve todo el análisis."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        # Cargar dataframe directamente del stream
        # Asumimos formato mHealth
        df = pd.read_csv(file, header=None, sep=r'\t', engine='python')
        
        # Validación básica de dimensiones
        if df.shape[1] < 24:
             return jsonify({"error": f"El archivo tiene {df.shape[1]} columnas, se requieren al menos 24 (23 features + label)."}), 400
        
        if df.shape[1] > 24:
             df = df.iloc[:, :24]

        df.columns = FEATURE_COLS + ['label']
        # Eliminar las filas con label=0 al igual que load_subject_data
        df = df[df['label'] != 0].copy()
        
        # Feature Engineering (Magnitudes)
        df['mag_acc_ankle'] = np.sqrt(df['acc_ankle_x']**2 + df['acc_ankle_y']**2 + df['acc_ankle_z']**2)
        df['mag_gyro_ankle'] = np.sqrt(df['gyro_ankle_x']**2 + df['gyro_ankle_y']**2 + df['gyro_ankle_z']**2)
        df['mag_acc_arm'] = np.sqrt(df['acc_arm_x']**2 + df['acc_arm_y']**2 + df['acc_arm_z']**2)
        df['mag_gyro_arm'] = np.sqrt(df['gyro_arm_x']**2 + df['gyro_arm_y']**2 + df['gyro_arm_z']**2)
        
        # Ventaneo
        X_test, y_test = create_windows(df, WINDOW_SIZE, OVERLAP)
        
        if len(X_test) == 0:
            return jsonify({"error": "No se pudieron generar ventanas válidas (datos insuficientes o vacíos)."}), 400
            
        # Predicción
        y_pred = predict_activities(X_test)
        
        if y_pred is None:
            return jsonify({"error": "Modelo no cargado en el servidor."}), 500
            
        step_duration = (WINDOW_SIZE - OVERLAP) / SAMPLING_RATE
        
        # Generar todos los datos
        stats_data = generate_prediction_stats(y_test, y_pred, step_duration)
        timeline_data = generate_timeline_data(y_test, y_pred, step_duration)
        matrix_data = generate_confusion_matrix_data(y_test, y_pred)
        
        return jsonify({
            "status": "success",
            "filename": file.filename,
            "prediction_stats": stats_data,
            "timeline_data": timeline_data,
            "matrix_data": matrix_data
        })
        
    except Exception as e:
        print(f"Error procesando archivo subido: {e}")
        return jsonify({"error": f"Error procesando archivo: {str(e)}"}), 500

@app.route('/api/timeline/<int:subject_id>', methods=['GET'])
def get_timeline(subject_id):
    """Obtiene timeline de actividades para visualización."""
    df = load_subject_data(subject_id)
    if df is None:
        return jsonify({"error": f"Sujeto {subject_id} no encontrado"}), 404
    
    X_test, y_test = create_windows(df, WINDOW_SIZE, OVERLAP)
    y_pred = predict_activities(X_test)
    
    if y_pred is None:
        return jsonify({"error": "Modelo no cargado"}), 500
    
    step_duration = (WINDOW_SIZE - OVERLAP) / SAMPLING_RATE
    
    data = generate_timeline_data(y_test, y_pred, step_duration)
    data["subject_id"] = subject_id
    return jsonify(data)

@app.route('/api/confusion-matrix/<int:subject_id>', methods=['GET'])
def get_confusion_matrix(subject_id):
    """Retorna datos para la matriz de confusión."""
    df = load_subject_data(subject_id)
    if df is None:
        return jsonify({"error": f"Sujeto {subject_id} no encontrado"}), 404
    
    X_test, y_test = create_windows(df, WINDOW_SIZE, OVERLAP)
    y_pred = predict_activities(X_test)
    
    if y_pred is None:
        return jsonify({"error": "Modelo no cargado"}), 500
    
    data = generate_confusion_matrix_data(y_test, y_pred)
    data["subject_id"] = subject_id
    return jsonify(data)

if __name__ == '__main__':
    load_model()
    print("\n🚀 Servidor Backend HAR iniciando...")
    print("📍 URL: http://localhost:5000")
    print("📋 Endpoints disponibles:")
    print("   GET  /api/health")
    print("   GET  /api/activities")
    print("   GET  /api/subjects")
    print("   GET  /api/predict/<subject_id>")
    print("   POST /api/predict")
    print("   POST /api/predict-file")
    print("   GET  /api/timeline/<subject_id>")
    print("   GET  /api/confusion-matrix/<subject_id>\n")
    app.run(debug=True, port=5000)
