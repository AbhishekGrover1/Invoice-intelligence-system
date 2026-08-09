"""
Trains the freight cost regression model and saves the best performing
model (by MAE) to the project's shared `models/` directory.

Run from anywhere, e.g.:
    python freight_cost_prediction/train.py
    (or)  cd freight_cost_prediction && python train.py
"""
import sys
from pathlib import Path

import joblib

# Resolve paths relative to this file's location (not the current working
# directory) so the script behaves the same no matter where it's launched
# from -- this mirrors the fix applied to invoice_flagging/train.py and to
# the inference/ layer.
THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from modeling_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)


def main():
    db_path = BASE_DIR / "Data" / "inventory.db"

    # Single shared models/ directory at the project root -- this is also
    # where the inference layer (inference/predict_freight.py) expects to
    # find predict_freight_model.pkl.
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(exist_ok=True)

    # Load data
    df = load_vendor_invoice_data(str(db_path))

    # Prepare data
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate models
    results = []
    results.append(evaluate_model(lr_model, X_test, y_test, "Linear Regression"))
    results.append(evaluate_model(dt_model, X_test, y_test, "Decision Tree Regression"))
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest Regression"))

    # Select best model (lowest MAE)
    best_model_info = min(results, key=lambda x: x["mae"])
    best_model_name = best_model_info["model_name"]

    best_model = {
        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model
    }[best_model_name]

    # Save best model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: {model_path}")


if __name__ == "__main__":
    main()
