# bot_utils.py

from gemini_utils import MEDIAN_VALUES  # Import the MEDIAN_VALUES dictionary
from model_utils import get_model

# Load the model
model = get_model()

def predict_price(features):
    """Predict house price using the trained model."""
    # Ensure that features are in a numerical list format (with default values if missing)
    model_input = [
        features.get('MedInc', MEDIAN_VALUES['MedInc']),
        features.get('HouseAge', MEDIAN_VALUES['HouseAge']),
        features.get('AveRooms', MEDIAN_VALUES['AveRooms']),
        features.get('AveBedrms', MEDIAN_VALUES['AveBedrms']),
        features.get('Population', MEDIAN_VALUES['Population']),
        features.get('AveOccup', MEDIAN_VALUES['AveOccup']),
        features.get('Latitude', MEDIAN_VALUES['Latitude']),
        features.get('Longitude', MEDIAN_VALUES['Longitude'])
    ]

    # Ensure that model_input is a valid list with numbers
    print(f"Model Input: {model_input}")

    prediction = model.predict([model_input])[0]  # Predicting the house price
    return prediction * 100000  # Convert to actual price in INR
