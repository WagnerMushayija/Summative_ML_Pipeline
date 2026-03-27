# src/prediction.py
import tensorflow as tf
import numpy as np
from PIL import Image
import os

from .preprocessing import IMG_HEIGHT, IMG_WIDTH, get_class_names
from .model import load_trained_model


class ImagePredictor:
    def __init__(self, model_path='../models/intel_image_model.keras'):
        self.model = load_trained_model(model_path)
        self.class_names = get_class_names()
        print("ImagePredictor initialized successfully!")

    def predict_image(self, image_path):
        """Predict class for a single image file"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx] * 100

        predicted_class = self.class_names[predicted_class_idx]

        return {
            'class': predicted_class,
            'confidence': round(confidence, 2),
            'probabilities': {self.class_names[i]: round(float(predictions[0][i]) * 100, 2)
                              for i in range(len(self.class_names))}
        }

    def predict_from_array(self, image_array):
        """Predict from numpy array (useful for API/Streamlit)"""
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)

        predictions = self.model.predict(image_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx] * 100

        return {
            'class': self.class_names[predicted_class_idx],
            'confidence': round(confidence, 2),
            'probabilities': {self.class_names[i]: round(float(predictions[0][i]) * 100, 2)
                              for i in range(len(self.class_names))}
        }


# For quick testing
if __name__ == "__main__":
    predictor = ImagePredictor()

    # Test with an example image (change the path to a real image from your test set)
    test_image_path = "../data/test/buildings/10000.jpg"  # ← Change this if needed

    if os.path.exists(test_image_path):
        result = predictor.predict_image(test_image_path)
        print("\nPrediction Result:")
        print(f"Predicted Class: {result['class']}")
        print(f"Confidence: {result['confidence']}%")
    else:
        print("Test image not found. Update the test_image_path.")