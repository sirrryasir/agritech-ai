# AgriTech AI - Crop Recommendation Model

Machine learning model that recommends crops based on soil composition and environmental conditions.

## What It Does

Analyzes soil nutrients (Nitrogen, Phosphorus, Potassium), temperature, humidity, pH, and rainfall to predict the best crop to cultivate. Uses scikit-learn RandomForest and GaussianNB classifiers trained on agricultural datasets.

## How It Works

1. User enters soil/weather parameters in web form
2. Frontend sends data to Flask backend
3. Backend runs data through trained ML model
4. Returns recommended crop

## Tech Stack

Backend:
- Flask with CORS
- scikit-learn (RandomForest, GaussianNB classifiers)
- Pandas, NumPy for data processing
- joblib for model serialization
- Runs on port 5000

Frontend:
- Next.js 16
- React 19
- TypeScript
- TailwindCSS

ML:
- Model training with train.py
- Data preprocessing and scaling
- Uses GridSearchCV for hyperparameter tuning

## Input Parameters

Soil Composition:
- N: Nitrogen content (ppm)
- P: Phosphorus content (ppm)
- K: Potassium content (ppm)

Environmental:
- Temperature (Celsius)
- Humidity (%)
- pH level
- Rainfall (mm)

## Project Structure

```
agritech-ai/
├── backend/
│   ├── app.py           Flask API server
│   ├── train.py         Model training script
│   ├── utils.py         Prediction utilities
│   └── models/          Trained model files
├── frontend/
│   ├── app/
│   │   ├── page.tsx     Main prediction interface
│   │   └── layout.tsx
├── notebooks/           Jupyter notebooks
├── dataset/             Training data
└── requirements.txt
```

## Installation

### Backend

```bash
cd backend
pip install -r ../requirements.txt
python app.py
```

Backend runs on http://localhost:5000

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on http://localhost:3000

## API Endpoint

POST /predict
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.87,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9
}
```

Returns:
```json
{
  "status": "success",
  "prediction": "rice",
  "input_data": {...}
}
```

## Models

- RandomForestClassifier: Primary model
- GaussianNB: Comparison model
- Trained on agricultural crop-soil-weather dataset

## License

MIT
