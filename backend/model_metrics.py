"""
Static evaluation metrics for the two shipped models.

These numbers are not computed at request time -- they're the test-set
results produced by running the actual training pipelines in
`freight_cost_prediction/train.py` and `invoice_flagging/train.py`
(80/20 split, random_state=42, both reproducible from Data/inventory.db).
They're recorded here once so the API and README can both quote the same
real figures instead of drifting out of sync.

If you retrain the models on different/updated data, re-run the two
train.py scripts and update this file to match.
"""

DATASET_INFO = {
    "source_name": "Vendor Performance Analysis",
    "source_url": "https://www.kaggle.com/datasets/harshmadhavan/vendor-performance-analysis",
    "tables": ["purchases", "purchase_prices", "vendor_invoice", "begin_inventory", "end_inventory"],
    "vendor_invoice_rows": 5543,
    "purchases_rows": 2372474,
}

FREIGHT_MODEL = {
    "task": "regression",
    "algorithm": "Linear Regression",
    "target": "Freight",
    "features": ["Dollars"],
    "train_rows": 4434,
    "test_rows": 1109,
    "metrics": {
        "mae": 24.11,
        "rmse": 124.72,
        "r2_pct": 96.99,
    },
    "compared_algorithms": [
        {"name": "Linear Regression", "mae": 24.11, "rmse": 124.72, "r2_pct": 96.99, "selected": True},
        {"name": "Decision Tree Regression", "mae": 32.97, "rmse": 150.31, "r2_pct": 95.63, "selected": False},
        {"name": "Random Forest Regression", "mae": 26.13, "rmse": 134.79, "r2_pct": 96.48, "selected": False},
    ],
    "selection_rule": "lowest MAE on the held-out test set",
}

INVOICE_FLAG_MODEL = {
    "task": "classification",
    "algorithm": "Random Forest Classifier (max_depth=6)",
    "target": "flag_invoice",
    "features": [
        "invoice_quantity",
        "invoice_dollars",
        "Freight",
        "total_item_quantity",
        "total_item_dollars",
    ],
    "train_rows": 4434,
    "test_rows": 1109,
    "class_balance_pct": {"not_flagged": 66.62, "flagged": 33.38},
    "metrics": {
        "accuracy_pct": 77.55,
        "not_flagged": {"precision_pct": 74.49, "recall_pct": 99.86, "f1_pct": 85.33},
        "flagged": {"precision_pct": 99.27, "recall_pct": 35.42, "f1_pct": 52.21},
        "macro_f1_pct": 68.77,
        "weighted_f1_pct": 73.86,
    },
    "confusion_matrix": {
        "true_negative": 724,
        "false_positive": 1,
        "false_negative": 248,
        "true_positive": 136,
    },
    "feature_importance": {
        "total_item_dollars": 0.2968,
        "invoice_dollars": 0.2170,
        "total_item_quantity": 0.2099,
        "invoice_quantity": 0.1485,
        "Freight": 0.1278,
    },
    "compared_algorithms": [
        {"name": "Logistic Regression", "accuracy_pct": 65.46, "f1_flagged_pct": 1.54, "selected": False},
        {"name": "Decision Tree Classifier", "accuracy_pct": 70.78, "f1_flagged_pct": 27.35, "selected": False},
        {"name": "Random Forest Classifier", "accuracy_pct": 77.55, "f1_flagged_pct": 52.21, "selected": True},
    ],
    "selection_rule": "highest F1 score (flagged class) on the held-out test set",
    "notes": (
        "This model is tuned toward precision over recall for the flagged "
        "class: of the invoices it flags, ~99% genuinely warrant manual "
        "review, but it only catches ~35% of all invoices that should be "
        "flagged. Recall could likely be improved by re-introducing "
        "avg_receiving_delay (the strongest single predictor in "
        "exploratory testing, excluded here because it isn't reliably "
        "available at invoice time) or by relaxing the max_depth cap."
    ),
}
