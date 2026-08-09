"""Endpoints for the freight cost regression model."""
from fastapi import APIRouter, HTTPException

from backend.schemas import FreightPredictionRequest, FreightPredictionResponse
from inference.predict_freight import predict_freight_cost

router = APIRouter(prefix="/api/predict", tags=["Freight Cost"])


@router.post(
    "/freight-cost",
    response_model=FreightPredictionResponse,
    summary="Predict freight cost from invoice dollar amount",
)
def predict_freight(payload: FreightPredictionRequest) -> FreightPredictionResponse:
    try:
        result = predict_freight_cost({"Dollars": [payload.dollars]})
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Freight prediction failed: {exc}") from exc

    predicted = float(result["Predicted_Freight"].iloc[0])
    ratio_pct = round((predicted / payload.dollars) * 100, 3) if payload.dollars else 0.0

    return FreightPredictionResponse(
        dollars=payload.dollars,
        predicted_freight=predicted,
        freight_ratio_pct=ratio_pct,
    )
