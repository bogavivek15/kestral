# Memorandum

**To:** Ritu Deshpande, Head of D2C Operations  
**From:** Returns Risk Project Team  
**Date:** 25 October 2026  
**Subject:** Pre-dispatch returns risk — why confirmation calls beat automatic dispatch holds  

---

## 1. Decision & Recommended Strategy

Deploy the model as a **decision-support aid for human-reviewed confirmation calls**, not as an automated dispatch hold. 

- **Proposed operational cutoff:** **12.0% return probability** (derived from the ₹45 call cost and 35% pilot prevention rate).
- **Action for flagged standard orders:** Dispatch operations team makes a quick confirmation call (verifying shipping address, appliance variant, and delivery slot) while keeping the order moving toward on-time dispatch.
- **Action for Kestrel Shield members:** Shield members generate Kestrel's highest customer lifetime value. Never place unattended holds or friction on their orders; use courteous confirmation only if needed.
- **Strict policy:** Never automatically hold dispatch based on model scores alone.

---

## 2. Reconciling Return Costs: ₹600 vs. ₹1,150

Your initial estimate of ₹600 per return accounts only for basic reverse courier freight. Finance Controller Farhan Sheikh confirmed that the authoritative Operations Policy v4.1 figure is **₹1,150 all-in average cost per return**:
- Reverse courier pickup logistics: ~₹420
- Inbound receiving, technician QC, and electrical testing: ~₹280
- Repackaging, relabelling, and restocking overhead: ~₹190
- Inventory write-down / open-box refurbishment depreciation: ~₹260

Finance uses ₹1,150 in official reporting. Basing operational triage on ₹600 would undercount return costs and distort intervention economics.

---

## 3. The Model Metrics (Why 95% Accuracy is Misleading)

The raw export contained 11,155 records, which included 651 duplicate partner-feed re-imports. Deduplicating by keeping authoritative CRM records leaves **10,504 unique historical orders** with **1,200 returns (11.42% prevalence)**.

We validated chronologically by training on 8,378 orders (April 2025 – March 2026) and testing on the subsequent 2,126 orders (April – June 2026). Our regularized Logistic Regression pipeline achieved:
- **ROC-AUC:** **0.7827** (solid discrimination across risk thresholds).
- **Average Precision (PR-AUC):** **0.4076** (3.5× higher than the 11.52% random prevalence baseline).
- **Brier Score:** **0.0853** (strong probability calibration).
- **Accuracy at 0.50 cutoff:** **89.56%**.

> **Crucial context on the 95% accuracy target:** In a population where only 11.42% of orders are returned, a naive rule predicting *every order as non-returned* achieves **88.48% baseline accuracy** while detecting exactly zero returns (0% recall). Pushing accuracy to 95% on imbalanced data would require suppressing flagged orders, causing the system to miss the vast majority of real returns.

At our operational **12.0% threshold**:
- **Recall:** **62.04%** (captures 152 of the 245 actual returns in validation).
- **Precision:** **25.76%** (1 in every 3.88 called orders would otherwise have been returned).
- **Flag rate:** **27.75%** (590 calls across 2,126 validation orders).

---

## 4. Financial Economics & Break-Even Calculation

- **Cost of intervention:** ₹45 per completed outsourced call.
- **Return prevention rate:** 35% of would-be returns prevented (from spring pilot).
- **Expected benefit per call:** $\text{Probability} \times 35\% \times ₹1,150$.
- **Break-even formula:**
  $$\text{Break-Even Probability} = \frac{₹45}{0.35 \times ₹1,150} = \frac{₹45}{₹402.50} \approx 11.18\%$$
  The proposed 12.0% threshold provides a sensible operating margin above break-even.

### Monthly Projections at Kestrel's Volume (~700 orders/month)
Applying our validated flag rate (27.75%) and true return rate (7.15%) to a monthly run-rate of 700 orders:
- **Calls initiated:** ~194 calls/month (calling spend: $194 \times ₹45 = \mathbf{₹8,742}$).
- **Would-be returns contacted:** ~50 orders.
- **Returns prevented:** $50 \times 35\% = \mathbf{17.5\text{ returns/month}}$.
- **Gross operational savings:** $17.5 \times ₹1,150 = \mathbf{₹20,144}$.
- **Net monthly bottom-line benefit:** $₹20,144 - ₹8,742 = \mathbf{₹11,402\text{/month}}$ (~₹1.37 Lakhs annualized).

---

## 5. Why Automated Dispatch Holds Are Dangerous

Policy §7 documents that holding an order at dispatch for more than 24 hours causes customer cancellation in approximately **12% of cases**.
- Cancelling an order forfeits the **entire gross margin / contribution margin** (typically thousands of rupees per appliance), not just the ₹1,150 return handling cost.
- Forcing holds on 194 orders/month would trigger roughly 23 customer cancellations, causing estimated lost margin of ₹45,000–₹80,000/month—completely wiping out any returns savings and damaging customer trust.
- **Conclusion:** Calls must be conducted promptly while warehouse picking/packing continues normally.

---

## 6. Action Plan for Next Week

1. **Launch a 4-Week Call Pilot:** Set up a calling queue for orders scoring $\ge 12.0\%$. Verify address, delivery timing, and product model. Ship on schedule.
2. **Empirical Measurement:** Randomly allocate 20% of flagged orders to a holdout control group (no call) to measure true causal return prevention under current customer conditions.
3. **Data Quality Fixes:** Notify engineering to fix the October 2025 payment gateway 100× currency bug upstream, and mandate delivery pincode capture on the mobile app to eliminate default `000000` records.
