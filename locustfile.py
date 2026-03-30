# locustfile.py
from locust import HttpUser, task, between
import random
import os
from pathlib import Path

class ImagePredictionUser(HttpUser):
    wait_time = between(2, 6)

    def on_start(self):
        self.test_images = []
        test_dir = Path("data/test")

        if not test_dir.exists():
            print("ERROR: data/test folder not found!")
            return

        for class_folder in test_dir.iterdir():
            if class_folder.is_dir():
                for img in class_folder.glob("*.jpg"):
                    self.test_images.append(str(img))
                for img in class_folder.glob("*.jpeg"):
                    self.test_images.append(str(img))

        print(f"✅ Loaded {len(self.test_images)} test images for load testing.")

    @task(1)
    def root_test(self):
        """Test basic connectivity"""
        with self.client.get("/", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Root: {resp.status_code}")

    @task(3)
    def predict_image(self):
        if not self.test_images:
            return

        image_path = random.choice(self.test_images)
        filename = os.path.basename(image_path)

        with open(image_path, "rb") as f:
            files = {"file": (filename, f, "image/jpeg")}

            with self.client.post("/predict", files=files, catch_response=True, timeout=180) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    error_msg = response.text[:500] if response.text else "No body"
                    response.failure(f"HTTP {response.status_code} - {error_msg}")