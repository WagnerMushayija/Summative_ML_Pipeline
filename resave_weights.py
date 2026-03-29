# resave_weights2.py
import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'
import tensorflow as tf
import numpy as np

print("Loading model...")
model = tf.keras.models.load_model('models/intel_image_model.keras', compile=False)

# Verify it works with a real test image
from PIL import Image
import glob

test_images = glob.glob('data/test/*/*')[:5]
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

for img_path in test_images:
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(160, 160))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, 0)
    pred = model.predict(arr, verbose=0)
    true_label = img_path.split('\\')[-2]
    pred_label = class_names[pred.argmax()]
    print(f"True: {true_label:10} | Predicted: {pred_label:10} | Confidence: {pred.max()*100:.1f}%")

print("\nSaving weights...")
model.save_weights('models/intel_image_weights2.weights.h5')
print("✅ Done!")