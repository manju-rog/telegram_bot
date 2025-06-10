# test_bot_utils.py
from bot_utils import predict_price

# Example features (you'd normally extract this from user input)
features = [8.3, 45.0, 6.5, 3.5, 1000, 1.0, 37.5, -122.3]  # Replace with actual feature values
predicted_price = predict_price(features)
print(f"Predicted House Price: ₹{predicted_price:,.2f}")
