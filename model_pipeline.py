import joblib
import re
import pandas as pd
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

model_path = BASE_DIR / "attendance_model.joblib"
preprocessor_path = BASE_DIR / "preprocessor.joblib"

with open(model_path, "rb") as f:
    model = joblib.load(f)

with open(preprocessor_path, "rb") as f:
    preprocessor = joblib.load(f)

def predict_pipeline(app_input: dict) -> dict:
    """
    Predict attendance using the trained model pipeline.
    repeat the preprocessing steps used during training.
    Args:
        app_input (dict): Input features for prediction.

    Returns:
        dict: Predicted attendance.
    """
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([app_input])

    # Preprocess the input dataset
    input_processed = preprocessor.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_processed)

    return {"predicted_attendance": prediction[0]}