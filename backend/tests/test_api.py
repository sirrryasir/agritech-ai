"""
AgriTech AI — Test Suite
Tests for the Flask API and ML model prediction pipeline.
"""
import pytest
import json
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app
from utils import load_models, get_prediction, prepare_features


# =========================================================
# Fixtures
# =========================================================

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def valid_input():
    """Standard valid input data for crop prediction."""
    return {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9
    }


@pytest.fixture
def rice_input():
    """Input values known to predict rice."""
    return {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.9
    }


@pytest.fixture
def apple_input():
    """Input values typically associated with apple crops."""
    return {
        "N": 20,
        "P": 125,
        "K": 200,
        "temperature": 23.0,
        "humidity": 92.0,
        "ph": 6.0,
        "rainfall": 110.0
    }


# =========================================================
# API Health Check Tests
# =========================================================

class TestHealthCheck:
    """Tests for the GET / health check endpoint."""

    def test_health_check_returns_200(self, client):
        """Health check endpoint should return 200 status code."""
        response = client.get('/')
        assert response.status_code == 200

    def test_health_check_returns_success(self, client):
        """Health check should return status 'success'."""
        response = client.get('/')
        data = json.loads(response.data)
        assert data['status'] == 'success'

    def test_health_check_has_endpoints_info(self, client):
        """Health check should describe available endpoints."""
        response = client.get('/')
        data = json.loads(response.data)
        assert 'endpoints' in data
        assert 'POST /predict' in data['endpoints']


# =========================================================
# Prediction Endpoint Tests
# =========================================================

class TestPredictEndpoint:
    """Tests for the POST /predict endpoint."""

    def test_predict_returns_200(self, client, valid_input):
        """Valid prediction request should return 200."""
        response = client.post(
            '/predict',
            data=json.dumps(valid_input),
            content_type='application/json'
        )
        assert response.status_code == 200

    def test_predict_returns_success_status(self, client, valid_input):
        """Prediction response should have status 'success'."""
        response = client.post(
            '/predict',
            data=json.dumps(valid_input),
            content_type='application/json'
        )
        data = json.loads(response.data)
        assert data['status'] == 'success'

    def test_predict_returns_prediction(self, client, valid_input):
        """Prediction response should contain a 'prediction' field."""
        response = client.post(
            '/predict',
            data=json.dumps(valid_input),
            content_type='application/json'
        )
        data = json.loads(response.data)
        assert 'prediction' in data
        assert isinstance(data['prediction'], str)
        assert len(data['prediction']) > 0

    def test_predict_returns_input_data(self, client, valid_input):
        """Prediction response should echo the input data."""
        response = client.post(
            '/predict',
            data=json.dumps(valid_input),
            content_type='application/json'
        )
        data = json.loads(response.data)
        assert 'input_data' in data
        assert data['input_data']['N'] == valid_input['N']

    def test_predict_rice(self, client, rice_input):
        """Known rice-optimal conditions should predict rice."""
        response = client.post(
            '/predict',
            data=json.dumps(rice_input),
            content_type='application/json'
        )
        data = json.loads(response.data)
        assert data['status'] == 'success'
        # Rice is a common prediction for these values
        assert data['prediction'].lower() == 'rice'

    def test_predict_with_float_values(self, client):
        """Prediction should handle all float values correctly."""
        float_input = {
            "N": 90.5,
            "P": 42.3,
            "K": 43.7,
            "temperature": 20.879,
            "humidity": 82.123,
            "ph": 6.502,
            "rainfall": 202.956
        }
        response = client.post(
            '/predict',
            data=json.dumps(float_input),
            content_type='application/json'
        )
        assert response.status_code == 200


# =========================================================
# Error Handling Tests
# =========================================================

class TestErrorHandling:
    """Tests for API error handling."""

    def test_missing_all_fields(self, client):
        """Request with empty JSON should return 400."""
        response = client.post(
            '/predict',
            data=json.dumps({}),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'

    def test_missing_one_field(self, client):
        """Request missing a single required field should return 400."""
        incomplete_input = {
            "N": 90,
            "P": 42,
            "K": 43,
            "temperature": 20.8,
            "humidity": 82.0,
            "ph": 6.5
            # Missing: "rainfall"
        }
        response = client.post(
            '/predict',
            data=json.dumps(incomplete_input),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'rainfall' in data['message']

    def test_no_json_body(self, client):
        """Request with no JSON body should return an error status."""
        response = client.post('/predict', content_type='application/json')
        assert response.status_code in [400, 500]
        data = json.loads(response.data)
        assert data['status'] == 'error'

    def test_wrong_method_on_predict(self, client):
        """GET request to /predict should fail (only POST allowed)."""
        response = client.get('/predict')
        assert response.status_code == 405


# =========================================================
# Model Utility Tests
# =========================================================

class TestModelUtils:
    """Tests for model loading and prediction utilities."""

    def test_models_load_successfully(self):
        """Model files should load without errors."""
        load_models()
        from utils import scaler, label_encoder, model
        assert scaler is not None
        assert label_encoder is not None
        assert model is not None

    def test_model_has_correct_classes(self):
        """Label encoder should have 22 crop classes."""
        load_models()
        from utils import label_encoder
        assert len(label_encoder.classes_) == 22

    def test_prepare_features_returns_array(self):
        """prepare_features should return a numpy array."""
        load_models()
        test_data = {
            "N": 90, "P": 42, "K": 43,
            "temperature": 20.8, "humidity": 82.0,
            "ph": 6.5, "rainfall": 202.9
        }
        result = prepare_features(test_data)
        assert result is not None
        assert result.shape == (1, 7)

    def test_get_prediction_returns_string(self):
        """get_prediction should return a crop name string."""
        load_models()
        test_data = {
            "N": 90, "P": 42, "K": 43,
            "temperature": 20.8, "humidity": 82.0,
            "ph": 6.5, "rainfall": 202.9
        }
        result = get_prediction(test_data)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_prediction_is_valid_crop(self):
        """Prediction should be one of the known crop types."""
        load_models()
        from utils import label_encoder
        test_data = {
            "N": 90, "P": 42, "K": 43,
            "temperature": 20.8, "humidity": 82.0,
            "ph": 6.5, "rainfall": 202.9
        }
        result = get_prediction(test_data)
        assert result in label_encoder.classes_
