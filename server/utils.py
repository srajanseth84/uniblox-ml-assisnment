import pandas as pd
import numpy as np
import joblib
import pickle
import os
from fastapi import HTTPException
from mylogging import logger


# Load model and preprocessor once at startup
def load_model_and_preprocessor(model_path, preprocessor_path):
    try:
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        logger.info(f"Loading preprocessor from {preprocessor_path}")
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)

        logger.info("Model and preprocessor loaded successfully")
        return model, preprocessor
    except Exception as e:
        logger.error(f"Failed to load model or preprocessor: {str(e)}")
        raise RuntimeError(f"Failed to load model or preprocessor: {str(e)}")


def process_prediction(employee_data, model, preprocessor, request_id):
    """Process a single prediction request"""
    try:
        # Convert to DataFrame (required by preprocessor)
        input_df = pd.DataFrame([employee_data])
        input_df = input_df.drop(columns=["employee_id"], errors="ignore")
        logger.info(f"Request {request_id}: Processing prediction for employee data")

        # Pre-process binary columns first
        if "has_dependents" in input_df.columns:
            input_df["has_dependents"] = input_df["has_dependents"].map(
                {"Yes": 1, "No": 0}
            )

        # Preprocess data
        X_processed = preprocessor.transform(input_df)

        # Make prediction
        prediction = int(model.predict(X_processed)[0])

        # Get probability if available
        try:
            probability = float(model.predict_proba(X_processed)[0][1])
        except (AttributeError, IndexError):
            probability = 0.0

        response = {
            "prediction": prediction,
            "probability": probability,
            "enrolled": "Yes" if prediction == 1 else "No",
            "request_id": request_id,
        }

        logger.info(
            f"Request {request_id}: Prediction successful - {response['enrolled']}"
        )
        return response

    except Exception as e:
        logger.error(f"Request {request_id}: Prediction failed - {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


def process_batch_prediction(batch_data, model, preprocessor, request_id):
    """Process a batch prediction request"""
    try:
        # Convert batch to DataFrame
        input_df = pd.DataFrame(batch_data["employees"])
        input_df = input_df.drop(columns=["employee_id"], errors="ignore")

        logger.info(
            f"Request {request_id}: Processing batch prediction for {len(input_df)} employees"
        )

        # Pre-process binary columns first
        if "has_dependents" in input_df.columns:
            input_df["has_dependents"] = input_df["has_dependents"].map(
                {"Yes": 1, "No": 0}
            )

        # Preprocess data
        X_processed = preprocessor.transform(input_df)

        # Make predictions
        predictions = model.predict(X_processed).tolist()

        # Get probabilities if available
        try:
            probabilities = model.predict_proba(X_processed)[:, 1].tolist()
        except (AttributeError, IndexError):
            probabilities = [0.0] * len(predictions)

        # Prepare response
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append(
                {
                    "prediction": int(pred),
                    "probability": float(prob),
                    "enrolled": "Yes" if pred == 1 else "No",
                    "request_id": f"{request_id}-{i}",
                }
            )

        logger.info(
            f"Request {request_id}: Batch prediction successful for {len(results)} employees"
        )
        return {"predictions": results, "request_id": request_id}

    except Exception as e:
        logger.error(f"Request {request_id}: Batch prediction failed - {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
