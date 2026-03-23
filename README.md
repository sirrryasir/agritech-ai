# AgriTech AI - Crop Recommendation System

## Project Description
AgriTech AI is a full-stack Machine Learning project designed to recommend the most suitable crop to plant based on soil metrics (Nitrogen, Phosphorus, Potassium) and weather conditions (Temperature, Humidity, Rainfall). The project includes an ML model training pipeline, a Flask REST API, and a beautiful Next.js frontend dashboard.

## Dataset Details
- **Source**: Kaggle (Crop Recommendation Dataset)
- **Size**: 2,201 rows, 8 columns.
- **Features**: N (Nitrogen), P (Phosphorus), K (Potassium), temperature, humidity, ph, rainfall.
- **Target**: label (Crop type, e.g., rice, maize, chickpea, etc.)

## Algorithms Used
We trained two classification algorithms to solve this multi-class prediction problem:
1. **Random Forest Classifier**: Chosen for its high accuracy and robustness against overfitting on tabular data.
2. **Gaussian Naive Bayes**: Chosen as a fast, probabilistic baseline model.

*Random Forest achieved the highest accuracy (>99%) and is the deployed model.*

## Example Commands

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model
```bash
cd backend
python train.py
```
*(This will save scaler.joblib, label_encoder.joblib, and best_model.joblib to backend/models/)*

### 3. Run the API (Backend)
```bash
cd backend
python app.py
```
*(The Flask API will run on http://localhost:5000)*

### 4. Run the Frontend
```bash
cd frontend
npm install
npm run dev
```
*(The user interface will run on http://localhost:3000)*

## Example API Usage

You can test the API using `curl`:

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "N": 90,
           "P": 42,
           "K": 43,
           "temperature": 20.8,
           "humidity": 82.0,
           "ph": 6.5,
           "rainfall": 202.9
         }'
```

**Response:**
```json
{
  "input_data": {
    "K": 43,
    "N": 90,
    "P": 42,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9,
    "temperature": 20.8
  },
  "prediction": "rice",
  "status": "success"
}
```

## Results Summary
Both models performed exceptionally well on the dataset. Random Forest was selected as the final model due to its perfect classification metrics on the test set. The models were successfully deployed via a Flask API connected to a Next.js frontend for an interactive user experience.
