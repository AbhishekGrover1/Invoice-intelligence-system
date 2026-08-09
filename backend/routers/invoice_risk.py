"""Endpoints for the invoice manual-review risk classifier."""
import io

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File

from backend.schemas import (
    InvoiceRiskRequest,
    InvoiceRiskResponse,
    BatchInvoiceRiskResult,
    BatchInvoiceRiskResponse,
)
from inference.predict_invoice_flag import predict_invoice_flag, FEATURES

router = APIRouter(prefix="/api/predict", tags=["Invoice Risk"])

RISK_LABEL_FLAGGED = "Flagged for Manual Review"
RISK_LABEL_APPROVED = "Auto-Approved"
MAX_BATCH_ROWS = 5000


def _to_response(flagged: bool, probability: float) -> InvoiceRiskResponse:
    """probability is P(flagged); confidence is the model's confidence in
    whichever class it actually predicted."""
    confidence = probability if flagged else (1 - probability)
    return InvoiceRiskResponse(
        flagged=flagged,
        risk_label=RISK_LABEL_FLAGGED if flagged else RISK_LABEL_APPROVED,
        flag_probability=round(float(probability), 4),
        confidence_pct=round(float(confidence) * 100, 2),
    )


@router.post(
    "/invoice-risk",
    response_model=InvoiceRiskResponse,
    summary="Evaluate a single invoice's manual-review risk",
)
def predict_invoice_risk(payload: InvoiceRiskRequest) -> InvoiceRiskResponse:
    data = {
        "invoice_quantity": [payload.invoice_quantity],
        "invoice_dollars": [payload.invoice_dollars],
        "Freight": [payload.freight],
        "total_item_quantity": [payload.total_item_quantity],
        "total_item_dollars": [payload.total_item_dollars],
    }
    try:
        result = predict_invoice_flag(data)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Invoice risk prediction failed: {exc}") from exc

    flagged = bool(result["Predicted_Flag"].iloc[0])
    probability = float(result["Flag_Probability"].iloc[0])
    return _to_response(flagged, probability)


@router.post(
    "/invoice-risk/batch",
    response_model=BatchInvoiceRiskResponse,
    summary="Evaluate many invoices at once from an uploaded CSV",
)
async def predict_invoice_risk_batch(
    file: UploadFile = File(
        ...,
        description=(
            "CSV with columns: invoice_quantity, invoice_dollars, Freight, "
            "total_item_quantity, total_item_dollars"
        ),
    )
) -> BatchInvoiceRiskResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    # Normalize whitespace and the one case-insensitive column ("Freight")
    # without silently guessing at anything else.
    rename_map = {}
    for col in df.columns:
        stripped = col.strip()
        rename_map[col] = "Freight" if stripped.lower() == "freight" else stripped
    df = df.rename(columns=rename_map)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"CSV is missing required column(s): {', '.join(missing)}. "
                f"Required columns: {', '.join(FEATURES)}"
            ),
        )

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV has no data rows.")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"CSV has too many rows ({len(df)}). Limit is {MAX_BATCH_ROWS} rows per upload.",
        )
    if df[FEATURES].isnull().any().any():
        bad_rows = sorted((df.index[df[FEATURES].isnull().any(axis=1)] + 1).tolist())
        raise HTTPException(
            status_code=400,
            detail=f"CSV has missing values in required columns on row(s): {bad_rows[:20]}",
        )

    try:
        data = {col: df[col].tolist() for col in FEATURES}
        result = predict_invoice_flag(data)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Batch prediction failed -- check that all rows contain valid numbers: {exc}",
        ) from exc

    results = []
    flagged_count = 0
    for i, (flag, prob) in enumerate(zip(result["Predicted_Flag"], result["Flag_Probability"])):
        flagged = bool(flag)
        flagged_count += int(flagged)
        base = _to_response(flagged, float(prob))
        results.append(BatchInvoiceRiskResult(row=i + 1, **base.model_dump()))

    return BatchInvoiceRiskResponse(
        total_rows=len(df),
        flagged_count=flagged_count,
        auto_approved_count=len(df) - flagged_count,
        results=results,
    )
