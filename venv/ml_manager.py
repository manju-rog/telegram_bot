# ml_manager.py

import pandas as pd
import joblib
import os
import logging
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

MODEL_PATH = "house_price_model.joblib"
INITIAL_DATA_PATH = "housing_initial.csv"
FEATURE_NAMES = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

def get_model_and_data():
    """Loads the model and base data, training if necessary."""
    if not os.path.exists(MODEL_PATH):
        logger.info("No existing model found. Training initial model...")
        df_initial = pd.read_csv(INITIAL_DATA_PATH)
        
        # Initial model with 100 trees
        model = RandomForestRegressor(n_estimators=100, random_state=42, warm_start=True)
        
        X = df_initial[FEATURE_NAMES]
        y = df_initial["MedHouseVal"]
        model.fit(X, y)
        
        joblib.dump(model, MODEL_PATH)
        logger.info(f"Initial model trained with {model.n_estimators} trees and saved.")
    
    model = joblib.load(MODEL_PATH)
    df_base = pd.read_csv(INITIAL_DATA_PATH) # Used for getting mean values
    return model, df_base

def retrain_model(new_data_path):
    """
    Performs incremental learning on the model with new data.
    Uses warm_start=True to add more estimators without retraining from scratch.
    """
    try:
        if not os.path.exists(new_data_path):
            return "Error: New data file not found."

        logger.info(f"Loading existing model from {MODEL_PATH} for retraining.")
        model = joblib.load(MODEL_PATH)
        
        # Ensure warm_start is enabled
        model.warm_start = True
        
        current_n_estimators = model.n_estimators
        logger.info(f"Model currently has {current_n_estimators} estimators.")

        # Increase the number of estimators (trees) in the forest
        model.n_estimators += 50
        
        logger.info(f"Loading new data from {new_data_path}...")
        df_new = pd.read_csv(new_data_path)
        X_new = df_new[FEATURE_NAMES]
        y_new = df_new["MedHouseVal"]
        
        logger.info(f"Retraining model. Target estimators: {model.n_estimators}...")
        model.fit(X_new, y_new) # partial_fit is not available, so fit() with warm_start adds new trees
        
        joblib.dump(model, MODEL_PATH)
        
        success_message = (
            "Retraining complete!\n"
            f"Model updated from {current_n_estimators} to {model.n_estimators} trees."
        )
        logger.info(success_message)
        return success_message

    except Exception as e:
        logger.error(f"An error occurred during retraining: {e}")
        return f"An error occurred during retraining: {e}"

def predict_price(features):
    """Predicts price using the current model and provided features."""
    model, df_base = get_model_and_data()
    
    # Create a full input dictionary, filling missing values with the mean
    model_input = {feature: df_base[feature].mean() for feature in FEATURE_NAMES}
    model_input.update(features)
    
    df_input = pd.DataFrame([model_input], columns=FEATURE_NAMES)
    prediction = model.predict(df_input)[0]
    return prediction * 100000