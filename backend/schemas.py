"""Pydantic request/response models for the API."""
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


# --- Freight cost ----------------------------------------------------------

class FreightPredictionRequest(BaseModel):
    dollars: float = Field(
        ..., gt=0,
        description="Total invoice dollar amount.",
        examples=[18500],
    )


class FreightPredictionResponse(BaseModel):
    dollars: float
    predicted_freight: float = Field(description="Model's predicted freight cost, in dollars.")
    freight_ratio_pct: float = Field(description="predicted_freight / dollars, as a percentage.")


# --- Invoice risk ------------------------------------------------------------

class InvoiceRiskRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "invoice_quantity": 120,
                "invoice_dollars": 5400.0,
                "freight": 210.0,
                "total_item_quantity": 118,
                "total_item_dollars": 5390.0,
            }]
        }
    )

    invoice_quantity: int = Field(..., gt=0, description="Units billed on the invoice.")
    invoice_dollars: float = Field(..., gt=0, description="Dollar amount billed on the invoice.")
    freight: float = Field(..., ge=0, description="Freight/shipping cost on the invoice.")
    total_item_quantity: int = Field(..., gt=0, description="Total units received against the PO.")
    total_item_dollars: float = Field(..., gt=0, description="Total dollar amount received against the PO.")


class InvoiceRiskResponse(BaseModel):
    flagged: bool
    risk_label: str = Field(description='"Flagged for Manual Review" or "Auto-Approved".')
    flag_probability: float = Field(description="Model's estimated probability the invoice should be flagged (0-1).")
    confidence_pct: float = Field(description="Model's confidence in the predicted class, as a percentage.")


class BatchInvoiceRiskResult(InvoiceRiskResponse):
    row: int


class BatchInvoiceRiskResponse(BaseModel):
    total_rows: int
    flagged_count: int
    auto_approved_count: int
    results: List[BatchInvoiceRiskResult]


# --- Meta ---------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str
    freight_model_loaded: bool
    invoice_flag_model_loaded: bool


class ErrorResponse(BaseModel):
    detail: str
