# Kestrel Home — Returns Risk (Variant A)

A local, end-to-end prototype for **pre-dispatch return-risk scoring** and operational call triage. Built with Python, Pandas, NumPy, scikit-learn, and FastAPI.

> **Data Privacy Notice:** In compliance with Kestrel Operations Policy v4.1 §10, confidential customer data and raw operational files are kept strictly in `private_data/` and excluded from Git commits via `.gitignore`. Per-order predictions in `predictions.csv` remain available locally for evaluation.

---

## 1. What the Project Does

1. **Deduplication:** Evaluates and removes 651 duplicate partner-outlet re-import records, prioritizing authoritative CRM records (resulting in 10,504 unique historical orders).
2. **Defect-Immune Feature Engineering:** Reconstructs basket order value directly from catalogue list price, quantity, and discount, eliminating a known October 2025 payment-gateway 100× currency glitch.
3. **Leakage Prevention:** Strictly excludes post-dispatch logistics timestamps (`pickup_scheduled_at`) and post-dispatch service events (`last_service_event_type`) that do not exist at warehouse dispatch time.
4. **Prompt Injection Sanitization:** Neutralizes 4 adversarial prompt-injection payloads found in delivery notes using deterministic allowlisted prefix categorisation.
5. **Model Benchmarking:** Compares a Majority Non-Return Baseline, Logistic Regression (C=0.2), HistGradientBoostingClassifier, and RandomForestClassifier on an out-of-time chronological holdout (April–June 2026).
6. **Economics-Driven Operational Triage:** Sets a 12.0% probability threshold based on Ops Policy v4.1 break-even arithmetic (₹45 call cost / [35% effectiveness × ₹1,150 return cost] = 11.18%), triggering courteous confirmation calls rather than harmful dispatch holds.
7. **FastAPI & Dispatch UI:** Serves real-time single-order risk predictions with exact model signals, operational warnings, and a browser UI.

---

## 2. Quickstart (Clean Machine Setup)

### System Requirements
- Python 3.11, 3.12, or 3.13
- Windows, macOS, or Linux
- Zero paid API keys required

### Windows PowerShell

```powershell
# 1. Create and activate a clean virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Start the FastAPI application
uvicorn app:app --host 127.0.0.1 --port 8000
```

### macOS / Linux

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Start the FastAPI application
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` for the dispatch interface.  
Open `http://127.0.0.1:8000/docs` for the interactive Swagger API documentation.  
Health check endpoint: `http://127.0.0.1:8000/health`.

---

## 3. Reproducing Model Training & Predictions

The 8 source files belong in `private_data/`:
- `train.csv`
- `test_unlabelled.csv`
- `customers.csv`
- `products.csv`
- `ops-policy.pdf`
- `email-thread.txt`
- `README.txt`
- `sample_submission.csv`

To train the models, run chronological validation, regenerate evidence reports, and generate `predictions.csv`:

```powershell
python train_and_predict.py --data-dir private_data
```

To verify `predictions.csv` against the required competition schema:

```powershell
python verify_predictions.py --data-dir private_data
```

---

## 4. Running Automated Tests

The test suite covers feature extraction, data leakage prevention, input validation, unseen category handling, API contracts, and batch-vs-API inference agreement:

```powershell
pytest -v
```

All 14 tests run locally in under 6 seconds with zero external network dependencies.

---

## 5. API Usage

### Single-Order Prediction (`POST /predict`)

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @example_order.json
```

### Sample Response
```json
{
  "order_id": "DEMO-001",
  "score": 0.386124,
  "return_risk_percent": 38.61,
  "risk_band": "HIGHER_RISK",
  "suggested_action": "PRE_DISPATCH_CONFIRMATION_CALL",
  "action_detail": "Route for human-reviewed confirmation call; do not automatically stop dispatch.",
  "model_signals": [
    "Historical customer return rate increased the model score.",
    "Cash on delivery payment mode increased the model score.",
    "Robot vacuum family increased the model score."
  ],
  "operational_notes": [
    "No additional address/history quality warning was detected from the supplied fields."
  ],
  "assumption_based_economics_inr": {
    "return_handling_cost": 1150.0,
    "confirmation_call_cost": 45.0,
    "assumed_call_effectiveness": 0.35,
    "expected_gross_saving_if_called": 155.41,
    "expected_net_benefit_if_called": 110.41
  },
  "interpretation_note": "The score is a model estimate, not certainty or proof of causation. Rupee figures assume the policy's 35% call effectiveness applies to this order. Human review is required; dispatch is never automatically held."
}
```

---

## 6. Project Architecture

```
kestrel-returns-risk-own/
├── app.py                     # FastAPI service & prediction endpoint
├── feature_engineering.py     # Deterministic feature pipeline & sanitization
├── train_and_predict.py       # Training, chronological validation & scoring
├── verify_predictions.py      # Submission schema verification
├── example_order.json         # Sample payload for API testing
├── predictions.csv            # Final scored test predictions (2,096 rows)
├── requirements.txt           # Production dependencies
├── memo_to_ritu.md            # One-page executive memorandum
├── submission_form_draft.md   # Complete answers to assignment questions
├── CODE_WALKTHROUGH.md        # Technical architectural breakdown
├── RECORDING_SCRIPT.md        # 3-minute screen recording guide
├── DATA_HANDLING_DECISION.md  # Policy §10 compliance documentation
├── artifacts/
│   └── model.joblib           # Trained scikit-learn pipeline & metadata
├── evidence/
│   ├── data_quality_report.md # Comprehensive data quality audit
│   ├── validation_report.md   # Chronological validation & candidate comparison
│   ├── model_comparison.csv   # Metrics comparison across model families
│   ├── threshold_analysis.csv # Decision threshold metrics & monthly scenarios
│   ├── calibration_table.csv  # 10-decile probability calibration data
│   └── validation_metrics.json# Machine-readable validation results
├── static/
│   └── index.html             # Dispatch operations screen
├── private_data/              # Local assignment files (git-ignored)
└── tests/
    ├── test_api.py            # API endpoints & validation error tests
    ├── test_features.py       # Feature transformation & leakage tests
    └── test_pipeline_and_consistency.py # Schema & API/batch agreement tests
```

---

## 7. Key Findings & Deliverable Summary

- **Validated Generalization:** Out-of-time chronological validation (April–June 2026) achieved **0.7827 ROC-AUC**, **0.4076 Average Precision**, and **0.0853 Brier score**.
- **Financial Scenario:** At the 12.0% operational threshold, Kestrel achieves an estimated net benefit of **~₹11,402/month** across 700 monthly orders (194 calls costing ₹8,742 saving ₹20,144 in return handling).
- **Dispatch Hold Warning:** Automated dispatch holds are rejected because holding an order >24 hours causes ~12% customer cancellations, forfeiting full margin.
