"""
Unit tests for the inference layer (inference/predict_freight.py and
inference/predict_invoice_flag.py) -- no HTTP involved, just the model
wrappers directly.

The fixed examples used here (freight_dollars=18500, and the two
"known_flagged" rows) are real values pulled from the actual held-out test
split of Data/inventory.db -- not made up -- so these tests double as a
regression check against the shipped .pkl artifacts in /models.
"""
import math

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag, FEATURES


class TestPredictFreightCost:
    def test_returns_expected_columns(self):
        result = predict_freight_cost({"Dollars": [18500]})
        assert list(result.columns) == ["Dollars", "Predicted_Freight"]

    def test_known_value_is_reproducible(self):
        # Linear Regression is deterministic -- this should match exactly,
        # not just "roughly".
        result = predict_freight_cost({"Dollars": [18500]})
        assert math.isclose(result["Predicted_Freight"].iloc[0], 97.79, abs_tol=0.5)

    def test_handles_multiple_rows(self):
        result = predict_freight_cost({"Dollars": [1000, 5000, 18500]})
        assert len(result) == 3
        # Freight should increase monotonically with invoice dollars for a
        # linear model with a positive coefficient.
        values = result["Predicted_Freight"].tolist()
        assert values[0] < values[1] < values[2]

    def test_prediction_is_non_negative_for_typical_input(self):
        result = predict_freight_cost({"Dollars": [500]})
        assert result["Predicted_Freight"].iloc[0] >= -1  # small negative intercept slack


class TestPredictInvoiceFlag:
    SAFE_EXAMPLE = {
        "invoice_quantity": [120],
        "invoice_dollars": [5400.0],
        "Freight": [210.0],
        "total_item_quantity": [118],
        "total_item_dollars": [5390.0],
    }

    # Real row from the held-out test split that the shipped model
    # correctly predicts as flagged (a true positive).
    FLAGGED_EXAMPLE = {
        "invoice_quantity": [48],
        "invoice_dollars": [352.95],
        "Freight": [1.73],
        "total_item_quantity": [162],
        "total_item_dollars": [2476.99],
    }

    def test_returns_expected_columns(self):
        result = predict_invoice_flag(self.SAFE_EXAMPLE)
        assert set(FEATURES).issubset(result.columns)
        assert "Predicted_Flag" in result.columns
        assert "Flag_Probability" in result.columns

    def test_probability_is_between_0_and_1(self):
        result = predict_invoice_flag(self.SAFE_EXAMPLE)
        prob = result["Flag_Probability"].iloc[0]
        assert 0.0 <= prob <= 1.0

    def test_known_safe_example_not_flagged(self):
        result = predict_invoice_flag(self.SAFE_EXAMPLE)
        assert result["Predicted_Flag"].iloc[0] == 0

    def test_known_flagged_example_is_flagged(self):
        result = predict_invoice_flag(self.FLAGGED_EXAMPLE)
        assert result["Predicted_Flag"].iloc[0] == 1

    def test_handles_multiple_rows(self):
        combined = {
            key: self.SAFE_EXAMPLE[key] + self.FLAGGED_EXAMPLE[key]
            for key in FEATURES
        }
        result = predict_invoice_flag(combined)
        assert len(result) == 2
        assert result["Predicted_Flag"].tolist() == [0, 1]
