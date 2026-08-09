"""
Central configuration for the backend service.

Everything here is either a fixed project path (resolved relative to this
file, so it works the same regardless of the working directory the process
was launched from) or an environment-overridable setting with a sane
default -- no required environment variables, so `uvicorn backend.main:app`
works out of the box after `pip install -r requirements.txt`.
"""
import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"

# --- App metadata --------------------------------------------------------
APP_NAME = "Invoice Intelligence System API"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "Serves two production models trained on vendor procurement data: "
    "a freight cost regressor and an invoice manual-review risk classifier."
)

# --- CORS ----------------------------------------------------------------
# The frontend is served by this same process, so CORS isn't required for
# the bundled deployment -- but it's left open (configurable via env var)
# so the API can also be called from a separately hosted frontend or from
# a local `file://` / dev-server page during development.
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = ["*"] if _raw_origins.strip() == "*" else [
    origin.strip() for origin in _raw_origins.split(",") if origin.strip()
]

# --- Server ---------------------------------------------------------------
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))
