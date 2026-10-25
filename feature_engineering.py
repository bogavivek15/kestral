"""Feature engineering shared by model training and the prediction API.

Only information available at dispatch is used. Raw delivery notes are never
passed directly into the model; they are reduced to a small allow-listed category.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "discount_pct",
    "qty",
    "clean_order_value_inr",
    "list_price_inr",
    "warranty_months",
    "promised_delivery_days",
    "customer_prior_orders",
    "customer_prior_returns",
    "prior_return_rate",
    "negative_tenure_flag",
    "clean_tenure_days",
    "product_age_days",
    "order_hour",
    "order_dayofweek",
    "is_weekend",
    "is_pincode_missing",
]

CATEGORICAL_FEATURES = [
    "sales_channel",
    "payment_mode",
    "family",
    "sku",
    "state",
    "city",
    "pincode_prefix",
    "note_category",
    "shield_member",
    "is_gift",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Exact beginning-of-note templates are used deliberately. If an instruction is
# appended to a normal note, it remains classified by the normal template and
# the appended text never reaches the model or an LLM.
NOTE_TEMPLATES = [
    ("Fragile - handle with care", "FRAGILE"),
    ("Office address, weekdays only", "OFFICE_WEEKDAYS"),
    ("Third floor, no lift", "STAIRS_NO_LIFT"),
    ("Leave with security", "LEAVE_WITH_SECURITY"),
    ("Do not call, WhatsApp only", "WHATSAPP_ONLY"),
    ("Customer requested morning slot", "MORNING_SLOT"),
    ("Call before delivery", "CALL_BEFORE_DELIVERY"),
    ("Ring twice", "RING_TWICE"),
    ("Deliver after 6 pm", "AFTER_6PM"),
]


def categorize_delivery_note(note: Any) -> str:
    """Map free text to a bounded category without retaining arbitrary text."""
    if note is None or pd.isna(note) or not str(note).strip():
        return "MISSING"

    text = str(note).strip()
    for template, category in NOTE_TEMPLATES:
        if text.startswith(template):
            return category
    if text.startswith("Landmark:"):
        return "LANDMARK"
    if text.startswith("Deliver to neighbour"):
        return "NEIGHBOUR"
    if text.startswith("Gate code"):
        return "GATE_CODE"
    return "OTHER"


def _normalise_pincode(series: pd.Series) -> pd.Series:
    values = series.astype("string").fillna("000000").str.replace(r"\D", "", regex=True)
    values = values.where(values.str.len() > 0, "000000")
    return values.str[-6:].str.zfill(6)


def create_features(enriched: pd.DataFrame) -> pd.DataFrame:
    """Create the model feature frame from orders already joined to references."""
    d = enriched.copy()

    order_dt = pd.to_datetime(d["order_placed_at"], errors="coerce")
    signup_dt = pd.to_datetime(d["signup_date"], errors="coerce")
    launch_dt = pd.to_datetime(d["launch_date"], errors="coerce")

    raw_tenure_days = (order_dt - signup_dt).dt.total_seconds() / 86400.0
    d["negative_tenure_flag"] = (raw_tenure_days < 0).astype(int)
    d["clean_tenure_days"] = raw_tenure_days.clip(lower=0)
    d["clean_tenure_days"] = d["clean_tenure_days"].fillna(0)
    d["product_age_days"] = ((order_dt - launch_dt).dt.total_seconds() / 86400.0).clip(lower=0).fillna(0)
    d["order_hour"] = order_dt.dt.hour.fillna(12).astype(float)
    d["order_dayofweek"] = order_dt.dt.dayofweek.fillna(0).astype(float)
    d["is_weekend"] = (d["order_dayofweek"] >= 5).astype(int)

    # Reconstruct value from the catalogue. This avoids the known October 2025
    # payment-gateway export defect where order_value_inr was multiplied by 100.
    list_price = pd.to_numeric(d["list_price_inr"], errors="coerce")
    quantity = pd.to_numeric(d["qty"], errors="coerce").fillna(1).clip(lower=0)
    discount = pd.to_numeric(d["discount_pct"], errors="coerce").fillna(0).clip(lower=0, upper=100)
    d["clean_order_value_inr"] = (list_price.fillna(0) * quantity * (1.0 - discount / 100.0)).astype(float)

    prior_orders = pd.to_numeric(d["customer_prior_orders"], errors="coerce").fillna(0).clip(lower=0)
    prior_returns = pd.to_numeric(d["customer_prior_returns"], errors="coerce").fillna(0).clip(lower=0)
    d["prior_return_rate"] = np.divide(
        prior_returns.to_numpy(dtype=float),
        prior_orders.to_numpy(dtype=float),
        out=np.zeros(len(d), dtype=float),
        where=prior_orders.to_numpy(dtype=float) > 0,
    )

    pincode = _normalise_pincode(d["delivery_pincode"])
    d["is_pincode_missing"] = (pincode == "000000").astype(int)
    d["pincode_prefix"] = pincode.str[:2].fillna("00")
    d["note_category"] = d["delivery_note"].apply(categorize_delivery_note)

    # Return only whitelisted, dispatch-available features. The ID columns,
    # source, pickup_scheduled_at, last_service_event_type, and raw note are out.
    for col in NUMERIC_FEATURES:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        if col not in d.columns:
            d[col] = "UNKNOWN"
        d[col] = d[col].astype("string").fillna("UNKNOWN").astype(str)

    return d.reindex(columns=FEATURE_COLUMNS).copy()


def join_reference_tables(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Join reference tables and create model features; fail loudly on bad joins."""
    if customers["customer_id"].duplicated().any():
        raise ValueError("customers.csv must contain one row per customer_id")
    if products["sku"].duplicated().any():
        raise ValueError("products.csv must contain one row per sku")

    enriched = orders.merge(customers, on="customer_id", how="left", validate="many_to_one")
    enriched = enriched.merge(products, on="sku", how="left", validate="many_to_one")
    required_reference = ["city", "state", "signup_date", "shield_member", "family", "list_price_inr", "warranty_months", "launch_date"]
    missing_counts = {col: int(enriched[col].isna().sum()) for col in required_reference}
    if any(missing_counts.values()):
        raise ValueError(f"Missing reference fields after joining customers/products: {missing_counts}")
    return create_features(enriched)


def features_from_context(order: dict[str, Any], customer: dict[str, Any], product: dict[str, Any]) -> pd.DataFrame:
    """Create one-row features for the API from an order plus customer/product context."""
    row = dict(order)
    row.update(customer)
    row.update(product)
    return create_features(pd.DataFrame([row]))
