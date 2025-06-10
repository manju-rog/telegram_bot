# model_utils.py

import os
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "house_price_model.joblib"

def get_model():
    if not os.path.exists(MODEL_PATH):
        # Train & save
        data = fetch_california_housing(as_frame=True)
        X, y = data.data, data.target
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(Xtr, ytr)
        joblib.dump(model, MODEL_PATH)
    # Load existing
    return joblib.load(MODEL_PATH)
