# Summative_ML_Pipeline

# Intel Image Classification MLOps Pipeline

A complete Machine Learning pipeline for classifying images into 6 categories (buildings, forest, glacier, mountain, sea, street) using MobileNetV2 transfer learning.

**Features:**
- Data acquisition & preprocessing
- Model training + fine-tuning (MobileNetV2)
- Single image prediction
- Bulk upload + retraining trigger
- Visualizations with interpretations (in Streamlit)
- Docker deployment
- Load testing with Locust

## Project Structure
Summative_ML_Pipeline/
├── README.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── main.py                 # FastAPI backend
├── app.py                  # Streamlit frontend
├── locustfile.py           # Load testing
├── keep_warm.py
├── data/
│   ├── train/
│   └── test/
├── models/
│   └── intel_image_weights2.weights.h5
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   └── retrainer.py
└── notebook/               


#### 4. Setup Instructions (Clear & Numbered)

```markdown
## Local Setup

1. Clone the repo
   ```bash
   git clone <your-repo-url>
   cd Summative_ML_Pipeline 
2. Build and run with DockerBash
    docker-compose up --build
```
### Access the app:
Streamlit UI: http://localhost:8501
FastAPI: http://localhost:8000


#### 5. Load Testing Results (This is key for the assignment)

```markdown
## Load Testing with Locust

**Tested on Render API (`https://summative-api.onrender.com`)**

- Users: 10
- RPS: ~1.2
- Failures: 0%
- Median latency: 2500 ms

![image 1](C:\Users\LENOVO\Pictures\Screenshots\Screenshot 2026-03-31 121829.png)
![image 2](C:\Users\LENOVO\Pictures\Screenshots\Screenshot 2026-03-31 122108.png)
![imag 3](C:\Users\LENOVO\Pictures\Screenshots\Screenshot 2026-03-31 122311.png)
![image 4](C:\Users\LENOVO\Pictures\Screenshots\Screenshot 2026-03-31 1234042.png)


**Note:** Render free tier shows higher latency due to cold starts. With keep-warm script, performance is stable.
```

## 6. Deployment

- **Backend API**: Deployed on Render → https://summative-api.onrender.com
- **Frontend**: Streamlit (can be deployed separately on Render)
- **Docker**: Fully containerized with `docker-compose.yml`
