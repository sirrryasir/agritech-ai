import joblib
import pandas as pd
import os

# Define the global objects
scaler = None
label_encoder = None
model = None

def load_models():
    """Load the scaler, label encoder, and pre-trained model."""
    global scaler, label_encoder, model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    encoder_path = os.path.join(models_dir, "label_encoder.joblib")
    model_path = os.path.join(models_dir, "best_model.joblib")
    
    if os.path.exists(scaler_path) and os.path.exists(encoder_path) and os.path.exists(model_path):
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(encoder_path)
        model = joblib.load(model_path)
        print("Models loaded successfully.")
    else:
        print("Warning: Model files not found. Please run train.py first.")

def prepare_features(data_dict):
    """
    Format standard dictionary JSON input into a scaled numpy array
    Expected features: N, P, K, temperature, humidity, ph, rainfall
    """
    if scaler is None:
        raise ValueError("Models not loaded properly.")
    
    # Needs to match exactly the pandas columns from training
    expected_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    df = pd.DataFrame([data_dict])[expected_cols]
    
    scaled_data = scaler.transform(df)
    return scaled_data

def get_prediction(data_dict):
    """Return the decoded string prediction from the model."""
    if model is None or label_encoder is None:
        raise ValueError("Models not loaded properly.")
        
    scaled_x = prepare_features(data_dict)
    prediction_encoded = model.predict(scaled_x)
    
    # decode the integer back to string (e.g., 'rice')
    predicted_crop = label_encoder.inverse_transform(prediction_encoded)[0]
    return predicted_crop
