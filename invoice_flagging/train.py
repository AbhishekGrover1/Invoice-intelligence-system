"""
Trains the invoice manual-approval risk classifier and saves the best
performing model (by F1 score) to the project's shared `models/` directory.

Run from anywhere, e.g.:
    python invoice_flagging/train.py
    (or)  cd invoice_flagging && python train.py
"""
import sys
from pathlib import Path

import joblib

# Make both "run from repo root" and "run from inside this folder" work the
# same way, and resolve the model output directory relative to the repo
# root rather than the current working directory.
THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from data_preprocessing import (
    load_invoice_data,
    apply_labels,
    prepare_features,
    split_data,
    scale_features
)
from modeling_evaluation import (
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"


def main():
    # Single shared models/ directory at the project root -- this is also
    # where the inference layer (inference/predict_invoice_flag.py) expects
    # to find predict_flag_invoice.pkl and scaler.pkl.
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(exist_ok=True)

    # Load data
    df = load_invoice_data()
    df = apply_labels(df)

    # Prepare data
    X, y = prepare_features(df, FEATURES, TARGET)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, model_dir / "scaler.pkl"
    )

    # Train models
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    dt_model = train_decision_tree(X_train_scaled, y_train)
    rf_model = train_random_forest(X_train_scaled, y_train)

    # Evaluate models
    results = []
    results.append(evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression"))
    results.append(evaluate_model(dt_model, X_test_scaled, y_test, "Decision Tree Classifier"))
    results.append(evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest Classifier"))

    # Select best model (highest F1 score)
    best_model_info = max(results, key=lambda x: x["f1"])
    best_model_name = best_model_info["model_name"]

    best_model = {
        "Logistic Regression": lr_model,
        "Decision Tree Classifier": dt_model,
        "Random Forest Classifier": rf_model
    }[best_model_name]

    # Save best model
    model_path = model_dir / "predict_flag_invoice.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: {model_path}")


if __name__ == "__main__":
    main()
