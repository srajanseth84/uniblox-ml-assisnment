from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime
from mylogging import logger, log_requests_middleware
from mymodels import (
    BatchEmployeeData,
    BatchPredictionResponse,
    EmployeeData,
    PredictionResponse,
)
from utils import (
    load_model_and_preprocessor,
    process_prediction,
    process_batch_prediction,
)

# Define paths
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model_random_forest.joblib")
PREPROCESSOR_PATH = os.environ.get(
    "PREPROCESSOR_PATH", "preprocessor/preprocessor_20250409_221944.pkl"
)

cache = {}

# Create FastAPI app
app = FastAPI(
    title="Insurance Enrollment Prediction API",
    description="API for predicting insurance enrollment likelihood",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
app.middleware("http")(log_requests_middleware)


# Loading model and preprocessor on startup
@app.on_event("startup")
async def get_model_and_preprocessor() -> None:
    """Load model and preprocessor on startup"""
    try:
        model_r, preprocessor_r = load_model_and_preprocessor(
            MODEL_PATH, PREPROCESSOR_PATH
        )
        cache["model"] = model_r
        cache["preprocessor"] = preprocessor_r
    except Exception as e:
        logger.error(f"Error loading model and preprocessor: {str(e)}")
        raise RuntimeError(f"Failed to load model and preprocessor: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Insurance Enrollment Prediction API", "status": "active"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL_PATH,
        "preprocessor_path": PREPROCESSOR_PATH,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    description="Predict insurance enrollment for a single employee",
)
async def predict(
    employee: EmployeeData,
    request: Request,
):
    try:
        request_id = request.state.request_id
        model = cache.get("model")
        preprocessor = cache.get("preprocessor")
        return process_prediction(employee.dict(), model, preprocessor, request_id)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    description="Batch prediction for multiple employees",
)
async def batch_predict(
    batch: BatchEmployeeData,
    request: Request,
):
    try:
        request_id = request.state.request_id
        model = cache.get("model")
        preprocessor = cache.get("preprocessor")
        print(batch.dict())
        return process_batch_prediction(batch.dict(), model, preprocessor, request_id)
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
