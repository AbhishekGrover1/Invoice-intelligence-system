# Invoice Intelligence System -- single-container deployment.
# Serves the FastAPI backend (/api/*) and the static frontend (/) from
# one process. No database is required at runtime -- only the trained
# model artifacts already committed under /models.

FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal on purpose -- scikit-learn/pandas wheels are
# manylinux, no compiler needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY inference/ ./inference/
COPY invoice_flagging/ ./invoice_flagging/
COPY freight_cost_prediction/ ./freight_cost_prediction/
COPY models/ ./models/
COPY frontend/ ./frontend/

ENV PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
