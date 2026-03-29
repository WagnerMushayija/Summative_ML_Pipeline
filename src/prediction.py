# src/prediction.py
import numpy as np
from PIL import Image
import os

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from .preprocessing import IMG_HEIGHT, IMG_WIDTH, get_class_names
from .model import load_trained_model


class ImagePredictor:
    def __init__(self, model_path='../models/intel_image_weights2.weights.h5'):
        self.model = load_trained_model(model_path)
        self.class_names = get_class_names()
        print("ImagePredictor initialized successfully!")

    def predict_image(self, image_path):
        """Predict class for a single image file"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img, dtype=np.float32)
        img_array = preprocess_input(img_array)  # ✅ applied here
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx] * 100

        return {
            'class': self.class_names[predicted_class_idx],
            'confidence': round(confidence, 2),
            'probabilities': {self.class_names[i]: round(float(predictions[0][i]) * 100, 2)
                              for i in range(len(self.class_names))}
        }

    def predict_from_array(self, image_array):
        """Predict from numpy array (useful for API/Streamlit)"""
        image_array = np.array(image_array, dtype=np.float32)
        image_array = preprocess_input(image_array)  # ✅ applied here
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


if __name__ == "__main__":
    predictor = ImagePredictor()
    test_image_path = "../data/test/buildings/10000.jpg"
    if os.path.exists(test_image_path):
        result = predictor.predict_image(test_image_path)
        print(f"Predicted: {result['class']} ({result['confidence']}%)")
    else:
        print("Test image not found.")