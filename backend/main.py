"""
FastAPI application entrypoint.

Run locally:
    uvicorn backend.main:app --reload

The API lives entirely under /api/*. Everything else is the static
frontend (frontend/index.html and friends), mounted last so it acts as a
catch-all without shadowing any API route.
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure the project root is importable (`inference`, `backend`, etc.)
# regardless of the working directory uvicorn was launched from.
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.model_metrics import DATASET_INFO, FREIGHT_MODEL, INVOICE_FLAG_MODEL
from backend.schemas import HealthResponse
from backend.routers import freight, invoice_risk
from inference.predict_freight import load_model as load_freight_model
from inference.predict_invoice_flag import load_model as load_flag_model, load_scaler as load_flag_scaler

# --- State: did the models actually load? ---------------------------------
_model_status = {"freight": False, "flag": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Fail fast and loudly if the model artifacts are missing/corrupt,
    instead of only discovering it on the first prediction request.
    """
    try:
        load_freight_model()
        _model_status["freight"] = True
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[startup] WARNING: freight model failed to load: {exc}")

    try:
        load_flag_model()
        load_flag_scaler()
        _model_status["flag"] = True
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[startup] WARNING: invoice flag model/scaler failed to load: {exc}")

    yield
    # No teardown needed -- models are just in-memory objects.


app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description=config.APP_DESCRIPTION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---------------------------------------------------------------
app.include_router(freight.router)
app.include_router(invoice_risk.router)


# --- Meta endpoints ----------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if all(_model_status.values()) else "degraded",
        version=config.APP_VERSION,
        freight_model_loaded=_model_status["freight"],
        invoice_flag_model_loaded=_model_status["flag"],
    )


@app.get("/api/models/info", tags=["Meta"])
def models_info() -> dict:
    """Real, reproducible test-set metrics for both shipped models. See
    backend/model_metrics.py for how these numbers were produced."""
    return {
        "dataset": DATASET_INFO,
        "freight_cost_model": FREIGHT_MODEL,
        "invoice_flag_model": INVOICE_FLAG_MODEL,
    }


# --- Static frontend (registered last -- see module docstring) -------------
app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
