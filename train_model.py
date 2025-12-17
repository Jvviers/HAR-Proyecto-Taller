import os
import sys
import json
import urllib.request
import zipfile
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# --- CONFIGURACIÓN ESTÉTICA ---
sns.set(style="whitegrid", palette="muted")

# --- CONFIGURACIÓN ---
DATASET_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip"
DATA_DIR = "MHEALTHDATASET"
PROCESSED_DIR = "output"
WINDOW_SIZE = 128  # 50Hz * 2.56 seg (Potencia de 2 para FFT rápida)
OVERLAP = 64       # 50%
PURITY_THRESHOLD = 0.70 # (90% de la ventana debe ser de la misma clase)
RANDOM_SEED = 42

# Estructura de columnas basada en documentación MHEALTH (23 features originales)
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

# Diccionario de Actividades con nombres en Inglés (Español)
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

# --- 1. DESCARGA Y EXTRACCIÓN ---
def download_and_extract():
    if not os.path.exists(DATA_DIR):
        print(f"Dataset no encontrado. Descargando de {DATASET_URL}...")
        zip_path = "mhealth.zip"
        try:
            urllib.request.urlretrieve(DATASET_URL, zip_path)
            print("Descomprimiendo...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            print("Descarga completada.")
        except Exception as e:
            print(f"Error durante la descarga o extracción: {e}")
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)
    else:
        print(f"Directorio '{DATA_DIR}' ya existe. Saltando descarga.")

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

# --- 2. PREPROCESAMIENTO E INGENIERÍA DE FEATURES (SEÑAL) ---
def load_subject_data(subject_id):
    """Carga y limpia datos de un sujeto específico."""
    # Los archivos se extraen típicamente en 'MHEALTHDATASET/mHealth_subjectX.log'
    # Ajustamos la ruta para ser robustos
    file_name = f"mHealth_subject{subject_id}.log"
    file_path = os.path.join(DATA_DIR, file_name)
    
    # Intento de fallback si hay una subcarpeta extra
    if not os.path.exists(file_path):
        file_path = os.path.join(DATA_DIR, "MHEALTHDATASET", file_name)

    if not os.path.exists(file_path):
        print(f"Advertencia: Archivo para sujeto {subject_id} no encontrado en {file_path}.")
        return None

    try:
        df = pd.read_csv(file_path, header=None, sep=r'\t', engine='python')
    except Exception as e:
        print(f"Error leyendo archivo {file_path}: {e}")
        return None

    # Asignar columnas (23 features + 1 label)
    df.columns = FEATURE_COLS + ['label']
    
    # Eliminar clase Null (0)
    df = df[df['label'] != 0].copy()
    
    # --- INGENIERÍA DE FEATURES: MAGNITUDES VECTORIALES ---
    # Se expande de 23 a 27 canales
    df['mag_acc_ankle'] = np.sqrt(df['acc_ankle_x']**2 + df['acc_ankle_y']**2 + df['acc_ankle_z']**2)
    df['mag_gyro_ankle'] = np.sqrt(df['gyro_ankle_x']**2 + df['gyro_ankle_y']**2 + df['gyro_ankle_z']**2)
    df['mag_acc_arm'] = np.sqrt(df['acc_arm_x']**2 + df['acc_arm_y']**2 + df['acc_arm_z']**2)
    df['mag_gyro_arm'] = np.sqrt(df['gyro_arm_x']**2 + df['gyro_arm_y']**2 + df['gyro_arm_z']**2)
    
    return df

def get_data_split():
    print("Cargando y procesando sujetos...")
    # Train: 1-6, Validation: 7-8
    train_subjects = [1, 2, 3, 4, 5, 6]
    val_subjects = [7, 8]
    
    train_dfs = [load_subject_data(i) for i in train_subjects]
    val_dfs = [load_subject_data(i) for i in val_subjects]
    
    # Filtrar Nones
    train_dfs = [d for d in train_dfs if d is not None]
    val_dfs = [d for d in val_dfs if d is not None]
    
    if not train_dfs or not val_dfs:
        raise ValueError("No se pudieron cargar los datos de entrenamiento o validación.")

    df_train = pd.concat(train_dfs, ignore_index=True)
    df_val = pd.concat(val_dfs, ignore_index=True)
    
    return df_train, df_val

# --- 3. EXTRACCIÓN DE CARACTERÍSTICAS (VENTANAS) ---
def calculate_energy(x):
    """Energía basada en FFT (suma de cuadrados de coeficientes)."""
    # Restar la media para eliminar componente DC
    fft_coeffs = np.fft.rfft(x - np.mean(x))
    return np.sum(np.abs(fft_coeffs)**2) / len(x)

def extract_features_from_window(window_data):
    """
    Entrada: Matriz (Window_Size, N_Channels) -> (100, 27)
    Salida: Vector aplanado (N_Channels * 7 stats) -> (189,)
    Stats: Mean, Std, Min, Max, Median, Skewness, Energy
    """
    # Estadísticas básicas vectorizadas
    # shape: (27,) cada una
    mean = np.mean(window_data, axis=0)
    std = np.std(window_data, axis=0)
    mins = np.min(window_data, axis=0)
    maxs = np.max(window_data, axis=0)
    medians = np.median(window_data, axis=0)
    skewness = stats.skew(window_data, axis=0)
    
    # Energía (FFT) - Iteramos por canal
    energies = np.apply_along_axis(calculate_energy, 0, window_data)
    
    # Concatenar todo
    # Orden final: [Mean_Ch1..Mean_Ch27, Std_Ch1..Std_Ch27, ..., Energy_Ch1..Energy_Ch27]
    features = np.concatenate([mean, std, mins, maxs, medians, skewness, energies])
    return features

def create_windows(df, window_size=100, overlap=50, purity_threshold=0.90):
    """Genera X (features) e y (labels) usando ventanas deslizantes con filtrado por pureza."""
    data_cols = [c for c in df.columns if c != 'label']
    X_raw = df[data_cols].values
    y_raw = df['label'].values
    
    X_windows = []
    y_windows = []
    
    step = window_size - overlap
    num_samples = X_raw.shape[0]
    
    print(f"Generando ventanas... (Signal length: {num_samples})")
    dropped_impure = 0
    
    for start in range(0, num_samples - window_size + 1, step):
        end = start + window_size
        
        # Segmento de ventana
        window_segment = X_raw[start:end, :]
        window_labels = y_raw[start:end]
        
        # Etiqueta: Moda
        mode_res = stats.mode(window_labels)
        
        # Compatibilidad scipy: extracción de scalar mode y count
        try:
            # Scipy moderno retorna arrays
            label_mode = mode_res.mode[0]
            count = mode_res.count[0]
        except (IndexError, TypeError, AttributeError):
            # Fallbacks para versiones antiguas o returns diferentes
            if np.ndim(mode_res.mode) == 0:
                label_mode = mode_res.mode
                count = mode_res.count
            else:
                 # Tuple fallback
                 label_mode = mode_res[0][0]
                 count = mode_res[1][0]
                 
        # --- FILTRADO POR PUREZA ---
        # Si la clase mayoritaria no cubre el umbral del tamaño de ventana, descartamos
        if count < (window_size * purity_threshold):
            dropped_impure += 1
            continue

        # Extracción de características
        features = extract_features_from_window(window_segment)
        
        X_windows.append(features)
        y_windows.append(label_mode)
        
    print(f"  Ventanas generadas: {len(X_windows)}")
    print(f"  Ventanas descartadas por impureza (<{int(purity_threshold*100)}%): {dropped_impure}")
    
    return np.array(X_windows), np.array(y_windows)

# --- 4. PIPELINE PRINCIPAL ---
def main():
    print("Iniciando Pipeline HAR MHEALTH...")
    
    # A. Descarga
    download_and_extract()
    
    # B. Carga y Feature Engineering (Magnitudes)
    df_train_raw, df_val_raw = get_data_split()
    
    print(f"Dimensiones Train (Signal): {df_train_raw.shape}")
    # Esperado: 23 originales + 4 magnitudes = 27 canales
    n_channels = len(df_train_raw.columns) - 1
    print(f"Canales detectados: {n_channels}")

    # C. Ventaneo y Extracción
    X_train, y_train = create_windows(df_train_raw, WINDOW_SIZE, OVERLAP, PURITY_THRESHOLD)
    X_val, y_val = create_windows(df_val_raw, WINDOW_SIZE, OVERLAP, PURITY_THRESHOLD)
    
    print(f"Ventanas Train: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Ventanas Val: {X_val.shape}, Labels: {y_val.shape}")
    
    # D. Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # E. Guardar Params Scaler
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_features_in": int(scaler.n_features_in_)
    }
    
    with open(os.path.join(PROCESSED_DIR, 'scaler_params.json'), 'w') as f:
        json.dump(scaler_params, f)
    print("Scaler params guardados.")

    # F. Entrenamiento
    print("Entrenando RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    # G. Evaluación
    print("\n---------- EVALUACIÓN EN TRAINING (1-6) ----------")
    y_train_pred = rf.predict(X_train_scaled)
    # Mapeo de actividades
    activity_labels = sorted(list(set(y_val) | set(y_train))) 
    target_names = [ACTIVITY_MAP.get(i, f"Unknown-{i}") for i in activity_labels]
    
    # Reporte resumido de train para no llenar la pantalla
    print(f"Accuracy Train: {np.mean(y_train_pred == y_train):.4f}")

    print("\n---------- EVALUACIÓN EN VALIDACIÓN (7-8) ----------")
    y_pred = rf.predict(X_val_scaled)
    # Mapeo de actividades
    activity_labels = sorted(list(set(y_val))) # Asegurar orden correcto de clases presentes
    target_names = [ACTIVITY_MAP[i] for i in activity_labels]
    
    print(classification_report(y_val, y_pred, target_names=target_names))
    
    # H. Matriz de Confusión
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', cbar=False,
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Matriz de Confusión - HAR MHEALTH')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'confusion_matrix.png'))
    print("Matriz de confusión guardada.")
    
    # I. Exportar Modelo a ONNX
    print("Exportando a ONNX...")
    model_path_onnx = os.path.join(PROCESSED_DIR, 'har_rf_model.onnx')
    
    # Definir tipo de entrada inicial: (None, n_features)
    # n_features debe coincidir con X_train_scaled.shape[1] (approx 189)
    initial_type = [('float_input', FloatTensorType([None, scaler.n_features_in_]))]
    
    # Especificamos target_opset=12 para compatibilidad con onnxruntime v1.14
    # Esto evita el error "Unsupported model IR version: 9"
    onnx_model = convert_sklearn(rf, initial_types=initial_type, target_opset=12)
    
    with open(model_path_onnx, "wb") as f:
        f.write(onnx_model.SerializeToString())
            
    print(f"Modelo ONNX exportado a: {model_path_onnx}")
    print(f"Features esperadas por el modelo: {scaler.n_features_in_}")

if __name__ == "__main__":
    main()