import os
from functools import lru_cache

import joblib
import pandas as pd

FEATURES = ["Dollars"]


@lru_cache(maxsize=1)
def load_model(model_path: str = None):
    """
    Load the trained freight cost regression model.

    Cached with lru_cache so the pickle is only read from disk once per
    process, instead of on every single prediction request.
    """
    if model_path is None:
        # Resolve relative to this file's location, then up one level to
        # the project root's shared `models/` directory. (This previously
        # pointed at a non-existent `inference/models/` folder.)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "..", "models", "predict_freight_model.pkl")
        model_path = os.path.abspath(model_path)
    return joblib.load(model_path)


def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.

    Parameters
    ----------
    input_data : dict
        Must contain the feature the model was trained on: Dollars

    Returns
    -------
    pd.DataFrame with predicted freight cost
    """
    model = load_model()

    input_df = pd.DataFrame(input_data)[FEATURES]
    input_df['Predicted_Freight'] = model.predict(input_df).round(2)
    return input_df


if __name__ == "__main__":
    # Example inference run (local testing)
    sample_data = {
        "Dollars": [18500, 9000]
    }
    prediction = predict_freight_cost(sample_data)
    print(prediction)
