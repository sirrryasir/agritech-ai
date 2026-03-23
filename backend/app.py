from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import load_models, get_prediction

app = Flask(__name__)
# Enable CORS for the Next.js frontend to communicate securely
CORS(app)

# Load the ML models on server startup
load_models()

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "success",
        "message": "AgriTech Crop Recommendation API is running.",
        "endpoints": {
            "POST /predict": {
                "expects": {
                    "N": "int",
                    "P": "int",
                    "K": "int",
                    "temperature": "float",
                    "humidity": "float",
                    "ph": "float",
                    "rainfall": "float"
                }
            }
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload provided."}), 400
        
        # Define required fields
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            return jsonify({
                "status": "error", 
                "message": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
            
        # Get Crop Prediction
        crop_name = get_prediction(data)
        
        return jsonify({
            "status": "success",
            "prediction": crop_name,
            "input_data": data
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
