"""
inference
=========
Thin, dependency-light prediction layer used by the FastAPI backend
(and usable standalone / from a notebook).

Loads the trained artifacts from the project's shared `models/` directory
and exposes one predict function per model:

    predict_freight.predict_freight_cost(...)
    predict_invoice_flag.predict_invoice_flag(...)
"""
