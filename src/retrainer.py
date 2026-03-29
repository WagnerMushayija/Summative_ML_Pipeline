# src/retrainer.py
import tensorflow as tf
from pathlib import Path
import streamlit as st

from .preprocessing import load_and_preprocess_data, get_data_augmentation
from .model import create_model, fine_tune_model


def retrain_model(epochs=8):
    """Full retraining pipeline"""
    try:
        BASE_DIR = Path(__file__).parent.parent.resolve()
        train_dir = str(BASE_DIR / "data" / "train")
        test_dir = str(BASE_DIR / "data" / "test")

        st.write("📥 Loading dataset...")

        train_ds, val_ds, test_ds = load_and_preprocess_data(
            train_dir=train_dir,
            test_dir=test_dir
        )

        st.write("🏗️ Creating new model...")
        model = create_model()

        # Stage 1: Train classifier head
        st.write("Stage 1: Training classifier head...")
        model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)

        # Stage 2: Fine-tuning
        st.write("Stage 2: Fine-tuning...")
        history = fine_tune_model(model, train_ds, val_ds, epochs=epochs)

        # Save the clean model
        model_path = BASE_DIR / "models" / "intel_image_model.keras"
        model.save(model_path, save_format='keras')

        st.success(f"✅ Model retrained successfully and saved cleanly!")
        st.info("The new model should now load properly in Docker.")

        return True, model

    except Exception as e:
        st.error(f"❌ Retraining failed: {str(e)}")
        return False, None