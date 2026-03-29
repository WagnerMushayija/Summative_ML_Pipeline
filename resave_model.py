# resave_h5.py
import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'
import tensorflow as tf

print("Loading original model...")
model = tf.keras.models.load_model('models/intel_image_model.keras', compile=False)
print("✅ Loaded!")

# Save as H5 - much more compatible format
model.save('models/intel_image_model.h5')
print("✅ Saved as intel_image_model.h5")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")