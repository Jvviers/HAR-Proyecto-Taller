# HAR Dashboard - Frontend + Backend

Proyecto de reconocimiento de actividades humanas (Human Activity Recognition) con modelo ONNX.

## Estructura del Proyecto

```
Proyecto_Taller/
├── backend/
│   ├── app.py              # API Flask
│   └── requirements.txt    # Dependencias Python
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   └── index.astro
│   │   ├── layouts/
│   │   │   └── Layout.astro
│   │   └── components/
│   │       ├── SubjectSelector.astro
│   │       └── ActivityCard.astro
│   ├── public/
│   ├── package.json
│   ├── astro.config.mjs
│   └── tailwind.config.mjs
├── output/
│   ├── har_rf_model.onnx   # Modelo entrenado
│   └── scaler_params.json  # Parámetros del scaler
└── MHEALTHDATASET/         # Datos de los sujetos
```

## Iniciar el Proyecto

### 1. Backend (Terminal 1)

```bash
cd backend

# Crear entorno virtual (opcional)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Iniciar servidor
python app.py
```

El backend estará disponible en: **http://localhost:5000**

### 2. Frontend (Terminal 2)

```bash
cd frontend

# Instalar dependencias
npm install

# Iniciar servidor de desarrollo
npm run dev
```

El frontend estará disponible en: **http://localhost:4321**

## API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/health` | Estado del servidor |
| GET | `/api/activities` | Lista de actividades |
| GET | `/api/subjects` | Sujetos disponibles |
| GET | `/api/predict/{id}` | Predicción para un sujeto |
| POST | `/api/predict` | Predicción con datos custom |
| GET | `/api/timeline/{id}` | Timeline de actividades |
| GET | `/api/confusion-matrix/{id}` | Matriz de confusión |

## Ejemplo de uso POST

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"sensor_data": [[...189 features...]]}'
```

## Tecnologías

- **Backend**: Flask, ONNX Runtime, NumPy, Pandas, SciPy
- **Frontend**: Astro, Tailwind CSS
- **Modelo**: Random Forest exportado a ONNX
