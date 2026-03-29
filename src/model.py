# src/model.py
import os
import tensorflow as tf
import keras
from tensorflow.keras import layers, models, applications

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


def load_trained_model(model_path='../models/intel_image_weights2.weights.h5'):
    """Build model architecture then load trained weights"""
    print(f"Loading weights from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Weights file not found at: {model_path}")

    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights=None
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(6, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    model.load_weights(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("✅ Model loaded with trained weights!")
    return model


def fine_tune_model(model, train_ds, val_ds, epochs=5):
    """Fine-tuning function"""
    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)
    return history