"""
invoice_flagging
=================
Training pipeline for the invoice manual-approval risk classifier.

Modules:
    data_preprocessing   -- loads + joins vendor_invoice/purchases data, builds the target label
    modeling_evaluation   -- trains and evaluates Logistic Regression / Decision Tree / Random Forest
    train                 -- CLI entrypoint that runs the full pipeline and saves the best model
"""
