# test_gemini_utils.py

from gemini_utils import extract_features_from_text

# Example query: "3 BHK house in Bangalore with 1500 sqft"
features = extract_features_from_text("3 BHK house in Bangalore with 1500 sqft")
print(features)
