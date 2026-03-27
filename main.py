# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pathlib import Path

from src.prediction import ImagePredictor

app = FastAPI(title="Intel Image Classifier API")

# Allow frontend to connect later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use absolute path to avoid path issues
BASE_DIR = Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / "models" / "intel_image_model.keras"

print(f"Looking for model at: {MODEL_PATH}")

# Initialize predictor with absolute path
predictor = ImagePredictor(model_path=str(MODEL_PATH))


@app.get("/")
def root():
    return {
        "message": "Intel Image Classification API is running!",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists()
    }


@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        temp_dir = BASE_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / file.filename

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Make prediction
        result = predictor.predict_image(str(temp_path))

        # Clean up temp file
        if temp_path.exists():
            os.remove(temp_path)

        return JSONResponse(content={
            "filename": file.filename,
            "prediction": result["class"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)