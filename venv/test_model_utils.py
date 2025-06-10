# test_model_utils.py

from model_utils import get_model

model = get_model()
print("✅ Model loaded. Sample prediction on zeros:", model.predict([[0]*8])[0])
