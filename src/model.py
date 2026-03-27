# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import os

from .preprocessing import IMG_HEIGHT, IMG_WIDTH, get_class_names


def create_model():
    """Create a fresh model (used for retraining)"""
    base_model = applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(len(get_class_names()), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_trained_model(model_path='../models/intel_image_model.keras'):
    """Load the saved model with better error handling"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    print(f"Loading model from: {model_path}")

    try:
        # Try loading with custom objects
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={
                'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input
            }
        )

        # Re-compile the model after loading
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("✅ Model loaded successfully!")
        print(f"Input shape: {model.input_shape}")
        return model

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\nTrying alternative loading method...")

        # Alternative: Load without the Lambda layer issue
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print("✅ Model loaded with alternative method!")
            return model
        except Exception as e2:
            print(f"Alternative also failed: {e2}")
            raise


def fine_tune_model(model, train_ds, val_ds, epochs=5):
    """Fine-tune the top layers of the model"""
    # Unfreeze the base model partially
    base_model = model.layers[0]  # MobileNetV2 is usually the first layer now
    base_model.trainable = True

    # Freeze all except the last 50 layers
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Starting fine-tuning...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)
    return history