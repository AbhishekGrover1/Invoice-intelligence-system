import os
from functools import lru_cache

import joblib
import pandas as pd

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]


@lru_cache(maxsize=1)
def load_model(model_path: str = None):
    """
    Load the trained invoice flagging classifier.

    Cached with lru_cache so the pickle is only read from disk once per
    process, instead of on every single prediction request.
    """
    if model_path is None:
        # Resolve relative to this file's location, then up one level to
        # the project root's shared `models/` directory. (This previously
        # pointed at a non-existent `inference/models/` folder.)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "..", "models", "predict_flag_invoice.pkl")
        model_path = os.path.abspath(model_path)
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def load_scaler(scaler_path: str = None):
    """
    Load the StandardScaler fitted during training. Cached for the same
    reason as load_model above.
    """
    if scaler_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        scaler_path = os.path.join(current_dir, "..", "models", "scaler.pkl")
        scaler_path = os.path.abspath(scaler_path)
    return joblib.load(scaler_path)


def predict_invoice_flag(input_data):
    """
    Predict whether new vendor invoices should be flagged for review.

    Parameters
    ----------
    input_data : dict
        Must contain the 5 features the model was trained on:
        invoice_quantity, invoice_dollars, Freight,
        total_item_quantity, total_item_dollars

    Returns
    -------
    pd.DataFrame with the predicted flag and the model's flag probability
    (probability of class 1, i.e. "should be flagged").
    """
    model = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)[FEATURES]

    # Same scaling the model was trained on - skipping this step
    # is what silently breaks predictions
    input_scaled = scaler.transform(input_df)
    input_scaled = pd.DataFrame(input_scaled, columns=FEATURES)

    input_df['Predicted_Flag'] = model.predict(input_scaled)
    input_df['Flag_Probability'] = model.predict_proba(input_scaled)[:, 1].round(4)
    return input_df


if __name__ == "__main__":
    # Example inference run (local testing)
    sample_data = {
        "invoice_quantity": [120, 45],
        "invoice_dollars": [5400.0, 1800.0],
        "Freight": [210.0, 65.0],
        "total_item_quantity": [118, 45],
        "total_item_dollars": [5390.0, 1750.0]
    }
    prediction = predict_invoice_flag(sample_data)
    print(prediction)
