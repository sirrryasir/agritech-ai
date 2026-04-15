# AgriTech AI -- Crop Recommendation System

> AI-powered crop recommendation system using Machine Learning.  
> **Final Project** -- Data Science & Machine Learning Bootcamp, Feb 2026

---

## Overview

**AgriTech AI** helps farmers make data-driven decisions by recommending the best crop to plant based on soil nutrients and weather conditions. The system uses a **Random Forest Classifier** trained on the Kaggle Crop Recommendation Dataset (2,200 samples x 22 crop types).

### Key Features
- **ML-powered predictions** -- Random Forest with GridSearchCV hyperparameter tuning (>99% accuracy)
- **Comprehensive EDA** -- correlation heatmaps, feature distributions, confusion matrices
- **Full-stack deployment** -- Flask REST API + Next.js frontend on Vercel
- **Tested** -- pytest test suite for API and model utilities

---

## Architecture

```
agritech-ai/
├── backend/
│   ├── app.py              <- Flask REST API (/predict endpoint)
│   ├── train.py            <- ML training pipeline (7-step workflow)
│   ├── utils.py            <- Model loading & prediction helpers
│   ├── models/             <- Saved model artifacts (.joblib)
│   │   ├── best_model.joblib
│   │   ├── scaler.joblib
│   │   ├── label_encoder.joblib
│   │   └── model_metadata.json
│   ├── eda_results/        <- Generated visualizations
│   │   ├── correlation_heatmap.png
│   │   ├── confusion_matrix_rf.png
│   │   ├── feature_importance.png
│   │   └── ... (10 charts total)
│   └── tests/
│       └── test_api.py     <- pytest test suite (20+ tests)
├── frontend/
│   └── app/
│       ├── page.tsx        <- Next.js main app (3 sections)
│       ├── layout.tsx      <- App layout + SEO
│       └── globals.css     <- Tailwind CSS
├── notebooks/
│   └── 01_eda_and_training.ipynb  <- Full EDA & training notebook
├── dataset/
│   └── Crop_recommendation.csv   <- Kaggle dataset (2,200 rows)
├── docs/
│   └── project_paper.md          <- Project documentation/paper
└── requirements.txt
```

---

## ML Pipeline

Following the bootcamp's 7-step ML workflow:

| Step | Description | Implementation |
|------|-------------|----------------|
| 1. Collect Data | Kaggle Crop Recommendation Dataset | 2,200 samples, 7 features, 22 crops |
| 2. Preprocess | Encoding + Scaling | LabelEncoder + StandardScaler |
| 3. Split | Train/Test partition | 80/20 stratified split |
| 4. Choose Model | Algorithm selection | Random Forest + Naive Bayes |
| 5. Train | Hyperparameter tuning | GridSearchCV (144 combos x 5-fold CV) |
| 6. Evaluate | Performance metrics | Accuracy, Classification Report, Confusion Matrix, K-Fold CV |
| 7. Deploy | Production API | Flask + Next.js + Vercel |

### Models Compared

| Model | Accuracy | Notes |
|-------|----------|-------|
| Gaussian Naive Bayes | 99.55% | Probabilistic baseline |
| **Random Forest (GridSearchCV)** | **99.55%** | **Selected for deployment** |

### Feature Importance
The most influential features for crop prediction:
1. **Potassium (K)** -- highest importance
2. **Humidity** -- weather condition impact
3. **Rainfall** -- annual precipitation

---

## Visualizations

The training pipeline generates 10 EDA visualizations:
- Crop Distribution
- Correlation Heatmap
- Feature Distributions (7 features)
- Feature Box Plots per Crop
- Confusion Matrix (Random Forest)
- Confusion Matrix (Naive Bayes)
- Model Accuracy Comparison
- Feature Importance Rankings
- K-Fold Cross Validation Scores

---

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend Setup

```bash
# Clone the repo
git clone https://github.com/sirrryasir/agritech-ai.git
cd agritech-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Train the model (generates all visualizations + model artifacts)
cd backend
python train.py

# Start the API server
python app.py
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Run Tests

```bash
cd backend
pytest tests/ -v
```

---

## API Reference

### Health Check
```
GET /
```
Returns API status and available endpoints.

### Predict Crop
```
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "N": 90,         
  "P": 42,         
  "K": 43,         
  "temperature": 20.8,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9
}
```

**Response:**
```json
{
  "status": "success",
  "prediction": "rice",
  "input_data": { ... }
}
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML/Data | Python, scikit-learn, pandas, NumPy, Matplotlib, Seaborn |
| Backend | Flask, Flask-CORS, joblib, Gunicorn |
| Frontend | Next.js 15, React, Tailwind CSS |
| Deployment | Vercel (Frontend), Flask API |
| Testing | pytest |
| Notebooks | Jupyter |

---

## Bootcamp Context

This project was built as the **final capstone project** for the [DS & ML Bootcamp (Feb 2026)](https://github.com/goobolabs/feb-ds-ml-bootcamp-2026), hosted by **GooboLabs** and sponsored by **Dugsiiye Online Courses**.

---

## License

This project is for educational purposes as part of the DS & ML Bootcamp 2026.
