"""
freight_cost_prediction
========================
Training pipeline for the freight cost regression model.

Modules:
    data_preprocessing   -- loads vendor_invoice data from SQLite
    modeling_evaluation   -- trains and evaluates Linear Regression / Decision Tree / Random Forest
    train                 -- CLI entrypoint that runs the full pipeline and saves the best model
"""
