"""FastAPI risk scoring API and small browser UI."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from feature_engineering import FEATURE_COLUMNS, features_from_context

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
PRIVATE_DATA_DIR = ROOT / "private_data"

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the pickled pipeline in the main process. joblib/loky can deadlock
    # if the first unpickle happens inside Uvicorn's request threadpool.
    load_artifact()
    yield


app = FastAPI(title="Kestrel Home — Pre-Dispatch Return Risk", version="1.0.0", lifespan=lifespan)
_ARTIFACT: dict[str, Any] | None = None
_CUSTOMERS: dict[str, dict[str, Any]] | None = None
_PRODUCTS: dict[str, dict[str, Any]] | None = None


class CustomerContext(BaseModel):
    city: str = "Pune"
    state: str = "MH"
    signup_date: str = "2024-02-11"
    shield_member: str = Field(default="N", pattern="^[YN]$")


class ProductContext(BaseModel):
    family: str = "Robot Vacuum"
    list_price_inr: float = Field(default=17599.0, ge=0)
    warranty_months: int = Field(default=12, ge=0)
    launch_date: str = "2024-03-06"


class OrderRequest(BaseModel):
    order_id: str = "DEMO-001"
    order_placed_at: str = "2026-07-01 12:40"
    customer_id: Optional[str] = None
    sku: str = "KH-RV-01"
    sales_channel: str = "app"
    payment_mode: str = "cod"
    discount_pct: float = Field(default=18, ge=0, le=100)
    qty: float = Field(default=1, ge=0)
    order_value_inr: Optional[float] = None  # accepted for compatibility, not used by model
    promised_delivery_days: int = Field(default=7, ge=0)
    delivery_pincode: str = "411001"
    is_gift: str = Field(default="N", pattern="^[YN]$")
    customer_prior_orders: float = Field(default=3, ge=0)
    customer_prior_returns: float = Field(default=1, ge=0)
    delivery_note: Optional[str] = "Call before delivery"
    # Supply these reference fields for self-contained requests. Otherwise the
    # API can look up IDs in private_data/customers.csv and products.csv.
    customer_context: Optional[CustomerContext] = None
    product_context: Optional[ProductContext] = None


def load_artifact() -> dict[str, Any] | None:
    global _ARTIFACT
    if _ARTIFACT is not None:
        return _ARTIFACT
    if MODEL_PATH.exists():
        _ARTIFACT = joblib.load(MODEL_PATH)
    return _ARTIFACT


def load_private_lookups() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    global _CUSTOMERS, _PRODUCTS
    if _CUSTOMERS is not None and _PRODUCTS is not None:
        return _CUSTOMERS, _PRODUCTS
    customer_path = PRIVATE_DATA_DIR / "customers.csv"
    product_path = PRIVATE_DATA_DIR / "products.csv"
    _CUSTOMERS = {}
    _PRODUCTS = {}
    if customer_path.exists():
        customers = pd.read_csv(customer_path).fillna("")
        _CUSTOMERS = customers.set_index("customer_id").to_dict(orient="index")
    if product_path.exists():
        products = pd.read_csv(product_path).fillna("")
        _PRODUCTS = products.set_index("sku").to_dict(orient="index")
    return _CUSTOMERS, _PRODUCTS


def _resolve_context(order: OrderRequest) -> tuple[dict[str, Any], dict[str, Any]]:
    customer: dict[str, Any] | None = order.customer_context.model_dump() if order.customer_context else None
    product: dict[str, Any] | None = order.product_context.model_dump() if order.product_context else None
    if customer is None or product is None:
        customers, products = load_private_lookups()
        if customer is None and order.customer_id and order.customer_id in customers:
            customer = customers[order.customer_id]
        if product is None and order.sku in products:
            product = products[order.sku]
    if customer is None:
        raise HTTPException(
            status_code=422,
            detail="Customer context is required when private_data/customers.csv is unavailable. Supply city, state, signup_date and shield_member in customer_context.",
        )
    if product is None:
        raise HTTPException(
            status_code=422,
            detail="Product context is required when private_data/products.csv is unavailable. Supply family, list_price_inr, warranty_months and launch_date in product_context.",
        )
    return customer, product


def _readable_feature_name(raw_name: str) -> str:
    name = raw_name.split("__", 1)[-1]
    aliases = {
        "customer_prior_returns": "Prior returns count",
        "customer_prior_orders": "Prior orders count",
        "prior_return_rate": "Historical customer return rate",
        "discount_pct": "Checkout discount",
        "promised_delivery_days": "Promised delivery time",
        "clean_order_value_inr": "Catalogue-adjusted basket value",
        "list_price_inr": "Catalogue list price",
        "warranty_months": "Warranty duration",
        "qty": "Order quantity",
        "clean_tenure_days": "Customer tenure",
        "negative_tenure_flag": "Signup date is after order date in source data",
        "product_age_days": "Product age at order",
        "order_hour": "Order hour",
        "order_dayofweek": "Day of week",
        "is_weekend": "Weekend order",
        "is_pincode_missing": "Default/missing delivery pincode",
    }
    if name in aliases:
        return aliases[name]
    categories = [
        "sales_channel", "payment_mode", "family", "sku", "state", "city",
        "pincode_prefix", "note_category", "shield_member", "is_gift",
    ]
    for category in categories:
        prefix = category + "_"
        if name.startswith(prefix):
            value = name[len(prefix):]
            labels = {
                "sales_channel": "Sales channel",
                "payment_mode": "Payment mode",
                "family": "Product family",
                "sku": "Product SKU",
                "state": "Customer state",
                "city": "Customer city",
                "pincode_prefix": "Pincode prefix",
                "note_category": "Delivery-note category",
                "shield_member": "Kestrel Shield membership",
                "is_gift": "Gift order",
            }
            return f"{labels[category]}: {value}"
    return name.replace("_", " ").strip().capitalize()


def _model_signals(artifact: dict[str, Any], x: pd.DataFrame, limit: int = 3) -> list[str]:
    """Return top positive linear contributions; not causal explanations."""
    model = artifact["model"]
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    transformed = preprocessor.transform(x)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    contributions = np.asarray(transformed[0]).ravel() * classifier.coef_[0]
    names = preprocessor.get_feature_names_out()
    positive = np.where(contributions > 1e-6)[0]
    if len(positive) == 0:
        return ["No single input feature strongly increased the score; the estimate reflects a combination of inputs."]
    top_idx = positive[np.argsort(contributions[positive])[::-1][:limit]]
    return [f"{_readable_feature_name(str(names[i]))} increased the model score." for i in top_idx]


def _operational_notes(order: OrderRequest, customer: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    pin = str(order.delivery_pincode).strip().zfill(6)
    if pin == "000000":
        notes.append("The delivery pincode is the default 000000 value; verify the address before dispatch.")
    if customer.get("shield_member") == "Y":
        notes.append("Kestrel Shield member: protect the high-lifetime-value relationship; use a courtesy confirmation call if selected and never automatically hold dispatch.")
    if int(order.customer_prior_returns) > int(order.customer_prior_orders):
        notes.append("Reported prior returns exceed prior orders; verify the customer-history fields.")
    if not notes:
        notes.append("No additional address/history quality warning was detected from the supplied fields.")
    return notes


@app.get("/health")
def health() -> dict[str, Any]:
    artifact = load_artifact()
    return {"status": "ok", "model_loaded": artifact is not None, "model_path_exists": MODEL_PATH.exists()}


@app.post("/predict")
def predict(order: OrderRequest) -> dict[str, Any]:
    artifact = load_artifact()
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model artifact is missing. Run `python train_and_predict.py --data-dir private_data` first.")
    customer, product = _resolve_context(order)
    row = order.model_dump(exclude={"customer_context", "product_context"})
    row["delivery_pincode"] = str(order.delivery_pincode)
    x = features_from_context(row, customer, product).reindex(columns=artifact["feature_columns"])
    score = float(artifact["model"].predict_proba(x)[:, 1][0])
    threshold = float(artifact["threshold"])
    is_shield = customer.get("shield_member") == "Y"
    if score >= threshold:
        action = "COURTESY_CONFIRMATION_CALL" if is_shield else "PRE_DISPATCH_CONFIRMATION_CALL"
        action_detail = "Route for human-reviewed confirmation call; do not automatically stop dispatch."
    else:
        action = "STANDARD_DISPATCH_REVIEW"
        action_detail = "No risk-triggered call recommended by this threshold; normal dispatch checks still apply."
    expected_gross_saving = score * float(artifact["call_prevention_rate"]) * float(artifact["return_cost_inr"])
    expected_net_benefit = expected_gross_saving - float(artifact["call_cost_inr"])
    return {
        "order_id": order.order_id,
        "score": round(score, 6),
        "return_risk_percent": round(score * 100, 2),
        "risk_band": "HIGHER_RISK" if score >= 0.30 else ("ELEVATED_RISK" if score >= threshold else "LOWER_RISK"),
        "suggested_action": action,
        "action_detail": action_detail,
        "model_signals": _model_signals(artifact, x),
        "operational_notes": _operational_notes(order, customer),
        "assumption_based_economics_inr": {
            "return_handling_cost": float(artifact["return_cost_inr"]),
            "confirmation_call_cost": float(artifact["call_cost_inr"]),
            "assumed_call_effectiveness": float(artifact["call_prevention_rate"]),
            "expected_gross_saving_if_called": round(expected_gross_saving, 2),
            "expected_net_benefit_if_called": round(expected_net_benefit, 2),
        },
        "interpretation_note": "The score is a model estimate, not certainty or proof of causation. Rupee figures assume the policy's 35% call effectiveness applies to this order. Human review is required; dispatch is never automatically held.",
    }


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    path = ROOT / "static" / "index.html"
    return path.read_text(encoding="utf-8")
