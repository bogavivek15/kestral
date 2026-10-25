# Validation & Model Selection Report — Kestrel Returns Risk

**Evaluation Window:** Out-of-Time Holdout (1 April 2026 to 30 June 2026, 2,126 orders)  
**Training Window:** Chronological Fit (1 April 2025 to 31 March 2026, 8,378 orders)  
**Target Class Prevalence:** 11.52% in validation (245 returns / 2,126 orders); 11.42% overall.

---

## 1. Candidate Model Comparison

Four models were compared on the exact same chronological holdout using identical pre-dispatch feature transformations:

| Model | ROC-AUC | PR-AUC (Avg Prec) | Brier Score | Accuracy @ 0.50 | Precision @ 0.12 | Recall @ 0.12 | F1 @ 0.12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression (C=0.2)** | **0.7827** | **0.4076** | **0.0853** | **89.56%** | **25.76%** | **62.04%** | **0.3641** |
| Logistic Regression (C=1.0) | 0.7822 | 0.4066 | 0.0855 | 89.46% | 26.58% | 61.63% | 0.3715 |
| Random Forest (depth=8) | 0.7635 | 0.3841 | 0.0902 | 88.52% | 21.56% | 68.98% | 0.3285 |
| HistGradientBoosting | 0.7517 | 0.3572 | 0.0891 | 89.13% | 26.80% | 60.82% | 0.3720 |
| Majority Non-Return Baseline | 0.5000 | 0.1152 | 0.1020 | 88.48% | 0.00% | 0.00% | 0.0000 |

### Why Logistic Regression was Selected
1. **Superior Ranking & Generalization:** Logistic Regression (C=0.2) achieves the highest ROC-AUC (0.7827) and PR-AUC (0.4076), beating complex tree ensembles across the temporal shift.
2. **Reliable Probability Calibration:** Lowest Brier score (0.0853), critical because probability estimates directly drive the financial break-even arithmetic.
3. **Transparent Explainability:** Model weights provide clean log-odds contributions for dispatch operator guidance without slow or unfaithful approximation wrappers.

---

## 2. Evaluation Metrics of Selected Model

| Metric | Result | Context / Meaning |
|---|---:|---|
| **ROC-AUC** | **0.7827** | Discrimination quality across all potential operational cutoffs. |
| **Average Precision (PR-AUC)** | **0.4076** | 3.5× higher than the 11.52% random prevalence baseline. |
| **Brier Score** | **0.0853** | Mean squared probability error (lower is better; baseline is 0.1020). |
| **Accuracy @ 0.50** | **89.56%** | Only 1.08% above the trivial majority-class baseline. |
| **Majority Baseline Accuracy** | **88.48%** | Accuracy achieved by predicting every order as non-returned (0% recall). |
| **Precision @ 0.12** | **25.76%** | 1 in 3.88 called orders would otherwise have been returned. |
| **Recall @ 0.12** | **62.04%** | Captures 152 of the 245 actual returns in the validation window. |
| **Flag Rate @ 0.12** | **27.75%** | 590 of 2,126 validation orders flagged for call review. |

### Confusion Matrices
- **At Standard 0.50 Threshold:**
  - True Negatives: 1,870 | False Positives: 11
  - False Negatives: 211 | True Positives: 34 (Recall: 13.88%)
  - *Demonstrates why standard 0.50 cutoff fails for operational risk triage on imbalanced data.*
- **At Operational 0.12 Triage Threshold:**
  - True Negatives: 1,443 | False Positives (courtesy calls): 438
  - False Negatives (unflagged returns): 93 | True Positives (caught returns): 152 (Recall: 62.04%)

---

## 3. Probability Calibration Across Deciles

| Score Bin | Total Orders | Mean Predicted Score | Observed Return Rate | Actual Returns |
|---|---:|---:|---:|---:|
| [0.007, 0.027] | 213 | 0.0197 | 0.0141 | 3 |
| (0.027, 0.041] | 213 | 0.0343 | 0.0282 | 6 |
| (0.041, 0.055] | 212 | 0.0478 | 0.0330 | 7 |
| (0.055, 0.070] | 213 | 0.0625 | 0.0563 | 12 |
| (0.070, 0.087] | 212 | 0.0780 | 0.0802 | 17 |
| (0.087, 0.108] | 213 | 0.0968 | 0.1033 | 22 |
| (0.108, 0.136] | 212 | 0.1215 | 0.1226 | 26 |
| (0.136, 0.180] | 213 | 0.1555 | 0.1690 | 36 |
| (0.180, 0.251] | 212 | 0.2104 | 0.2217 | 47 |
| (0.251, 0.776] | 213 | 0.3347 | 0.3239 | 69 |

Observed return rates track predicted probabilities monotonically with close decile calibration.

---

## 4. Reconciled Financial Economics

- **Policy Discrepancy Reconciled:** Ritu's initial email mentioned ₹600 return cost; Finance Controller Farhan Sheikh confirmed the official Ops Policy v4.1 figure of **₹1,150** all-in average cost (reverse logistics, QC inspection, repacking, and inventory write-down). ₹600 was an incomplete direct freight estimate.
- **Intervention Economics:**
  - Call cost: ₹45 per completed confirmation call.
  - Pilot effectiveness: 35% of returns prevented among called orders.
  - Break-Even Probability:
    `p* = Call Cost / (Effectiveness x Return Cost) = Rs 45 / (0.35 x Rs 1,150) = Rs 45 / Rs 402.50 = 11.18%`
  - Selected Operational Threshold: **12.0%** (break-even plus a modest operational buffer).

### Monthly Financial Projection (700 orders/month volume)
- Flagged orders: 700 x 27.75% = 194.25 calls/month.
- Calling expense: 194.25 x Rs 45 = Rs 8,741.25/month.
- Flagged true returns: 700 x 7.15% = 50.05 returns.
- Prevented returns: 50.05 x 35% = 17.52 returns.
- Gross savings: 17.52 x Rs 1,150 = Rs 20,144.40/month.
- **Net Monthly Benefit:** **Rs 11,403.15/month** (approx. Rs 11,402/month).

### Dispatch Hold Risk Evaluation
- Operations Policy §7 warns: *"An order held at dispatch for more than 24 hours is cancelled by the customer about 12% of the time."*
- A cancellation destroys the entire order contribution margin and customer goodwill.
- **Recommendation:** Never automatically freeze warehouse dispatch. Use model output exclusively to trigger courteous confirmation calls while keeping dispatch on schedule.

---

## 5. Limitations & Caveats

1. **Not Hidden Test Proof:** Chronological validation provides realistic out-of-time estimates, but cannot guarantee future hidden test performance.
2. **Exploratory Model Comparisons:** Comparison of multiple model families introduces slight selection optimism.
3. **Causal Attribution:** The 35% prevention rate is an operational assumption from the spring pilot; it must be verified prospectively via a controlled A/B trial.
4. **No Unattended Execution:** Predictions should support human dispatch review rather than automated order cancellation.
