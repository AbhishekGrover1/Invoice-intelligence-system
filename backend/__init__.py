"""
backend
=======
FastAPI service that wraps the two trained models (freight cost regression
and invoice risk classification) behind a small REST API, and serves the
static frontend from the same process.

Entrypoint: backend.main:app
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""
