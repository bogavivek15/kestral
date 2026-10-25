import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def base_payload():
    return {
        "order_id": "TEST-DEMO-1",
        "order_placed_at": "2026-07-01 12:40",
        "customer_id": "DEMO-CUSTOMER",
        "sku": "KH-RV-01",
        "sales_channel": "marketplace",
        "payment_mode": "cod",
        "discount_pct": 22,
        "qty": 1,
        "order_value_inr": 1234567,
        "promised_delivery_days": 8,
        "delivery_pincode": "411001",
        "is_gift": "N",
        "customer_prior_orders": 3,
        "customer_prior_returns": 2,
        "delivery_note": "Call before delivery",
        "customer_context": {
            "city": "Pune",
            "state": "MH",
            "signup_date": "2024-02-11",
            "shield_member": "N",
        },
        "product_context": {
            "family": "Robot Vacuum",
            "list_price_inr": 17599,
            "warranty_months": 12,
            "launch_date": "2024-03-06",
        },
    }


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_path_exists"] is True


def test_home_screen_serves():
    response = client.get("/")
    assert response.status_code == 200
    assert "Kestrel Home" in response.text
    assert "Order Dispatch Information" in response.text


def test_predict_returns_score_and_reasons():
    response = client.post("/predict", json=base_payload())
    assert response.status_code == 200, response.text
    result = response.json()
    assert 0 <= result["score"] <= 1
    assert result["order_id"] == "TEST-DEMO-1"
    assert len(result["model_signals"]) > 0
    assert result["risk_band"] in {"HIGHER_RISK", "ELEVATED_RISK", "LOWER_RISK"}
    assert result["suggested_action"] in {
        "PRE_DISPATCH_CONFIRMATION_CALL",
        "STANDARD_DISPATCH_REVIEW",
        "COURTESY_CONFIRMATION_CALL",
    }
    assert "automatically" in result["interpretation_note"].lower()
    assert "return_handling_cost" in result["assumption_based_economics_inr"]
    assert result["assumption_based_economics_inr"]["return_handling_cost"] == 1150.0
    assert result["assumption_based_economics_inr"]["confirmation_call_cost"] == 45.0


def test_predict_shield_member_courtesy_action():
    data = base_payload()
    data["customer_context"]["shield_member"] = "Y"
    # Ensure elevated risk so threshold is exceeded
    data["payment_mode"] = "cod"
    data["customer_prior_returns"] = 3
    data["customer_prior_orders"] = 3
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    if result["score"] >= 0.12:
        assert result["suggested_action"] == "COURTESY_CONFIRMATION_CALL"
    assert any("shield" in note.lower() for note in result["operational_notes"])


def test_predict_default_pincode_warning():
    data = base_payload()
    data["delivery_pincode"] = "000000"
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert any("000000" in note for note in result["operational_notes"])


def test_predict_with_private_reference_lookup():
    # When customer_id is a known ID in private_data/customers.csv, customer_context is optional
    data = base_payload()
    data["customer_id"] = "KC105196"
    data["sku"] = "KH-IC-03"
    data.pop("customer_context")
    data.pop("product_context")
    response = client.post("/predict", json=data)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["order_id"] == "TEST-DEMO-1"
    assert 0 <= result["score"] <= 1


def test_missing_context_for_unknown_customer_is_clear_error():
    data = base_payload()
    data["customer_id"] = "KC_NON_EXISTENT_99999"
    data.pop("customer_context")
    response = client.post("/predict", json=data)
    assert response.status_code == 422
    assert "Customer context is required" in response.text


def test_invalid_input_validation():
    # Discount > 100
    data = base_payload()
    data["discount_pct"] = 150
    res = client.post("/predict", json=data)
    assert res.status_code == 422

    # Negative quantity
    data = base_payload()
    data["qty"] = -1
    res = client.post("/predict", json=data)
    assert res.status_code == 422

    # Invalid is_gift
    data = base_payload()
    data["is_gift"] = "MAYBE"
    res = client.post("/predict", json=data)
    assert res.status_code == 422
