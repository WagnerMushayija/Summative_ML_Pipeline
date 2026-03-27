# test-run.py (updated)
from src.preprocessing import load_and_preprocess_data, get_class_names
from src.model import load_trained_model
from src.prediction import ImagePredictor

print("Class names:", get_class_names())

# Test model loading
model = load_trained_model()
print("Model loaded successfully!")

# Test predictor
predictor = ImagePredictor()

print("\n✅ All src/ modules imported and tested successfully!")