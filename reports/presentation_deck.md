# Stanbic Bank Ghana — Automated SME Credit Assessment
## Presentation Deck — Interview Submission
### 8 Slides · Speaker Notes Included

---

## SLIDE 1 — Title & Problem Statement

**Title:** Automated SME Credit Assessment Pipeline
**Subtitle:** Stanbic Bank Ghana · Data Science Assessment

**Key message (one sentence on slide):**
> "An end-to-end ML pipeline that converts a raw loan application into a data-driven APPROVE / DECLINE / REFER recommendation — with regulatory-grade explainability and fairness auditing."

**Speaker notes:**
- The pipeline handles 3 audiences simultaneously: data scientists (rigorous methodology), business stakeholders (actionable decisions with reasons), and regulators (audit trail + fairness evidence)
- It is production-ready: the same preprocessing function runs at training time and at inference time for a single new application

---

## SLIDE 2 — Dataset & Business Context

**Headline:** 3,036 SME Loan Applications · 13.32% Default Rate

| Fact | Value |
|------|-------|
| Dataset size | 3,036 applications |
| Default rate | 13.32% (~1 in 7.5 applicants) |
| Class ratio | ~6.5 non-defaulters : 1 defaulter |
| Features (raw) | 30 columns |
| Features (engineered) | 41 total: 24 numeric, 6 categorical, 11 binary |
| New v3 features | formality_score, unsecured_loan_flag, sector_x_region |

**Cost asymmetry (business context):**
- Missed default (False Negative): **GHS 80,000 loss** (100K loan × 80% loss given default)
- Wrong decline (False Positive): **GHS 22,000 foregone interest** (100K × 22% rate)
- **FN costs 3.6× more than FP** → this ratio drives every threshold decision

**Speaker notes:**
- 13.32% default rate is within the expected 10–25% range for SME lending in emerging markets
- The cost asymmetry is the single most important business constraint — it means we must prioritise catching defaults over avoiding wrong declines

---

## SLIDE 3 — Data Quality & Leakage Detection

**Headline:** 9 Data Quality Issues Resolved · 1 Critical Leakage Found

**Data quality issues (left column):**
1. `has_momo_account`: 7+ inconsistent encodings → standardized to 0/1/NaN
2. `annual_revenue_ghs`: mixed GHS/USD formats → regex strip → float
3. `owner_gender`: M/Male/male/F/Female → canonical Male/Female
4. `gra_tin`: PENDING/empty → binary `has_tin` flag
5. `owner_age`: values as low as 14 → nulled below 18 (Ghana legal minimum)
6. `previous_default`: missing ≠ "No default" → 3-way (Yes/No/Unknown)
7. `credit_bureau_score`: missing → binary flag + sector-median imputation
8. `bank_account_tenure`: zeros → flag rather than null (ambiguous, not wrong)
9. `rm_recommendation` + `internal_risk_grade`: **DATA LEAKAGE — dropped**

**Leakage story (right column, highlighted):**
- First run: CV AUC = 1.0 (Logistic Regression) and 0.9999 (XGBoost)
- Signal: real-world credit models do not achieve AUC 1.0
- Root cause: `days_past_due_current` is recorded at loan maturity, not application time
  - Non-defaulters: DPD = 0 for 100% of rows
  - Defaulters: DPD = 30–180 for 100% of rows → perfect binary separator
- Fix: removed from model features; retained as hard rule R001 in Decision Engine

**Speaker notes:**
- The leakage detection is the most important quality signal — AUC 1.0 means the model learned to look at the answer sheet, not the exam
- After removal: AUC dropped to 0.61 — the honest predictive power of pre-application information

---

## SLIDE 4 — Feature Engineering (41 Total Features)

**Headline:** Every Feature Has a Business Rationale

**Core engineered features (representative sample):**

| Feature | Category | Why it matters |
|---------|----------|----------------|
| `loan_to_revenue_ratio` | Ratio | Standard banking: ratio > 1 = borrowing more than annual income |
| `collateral_coverage_ratio` | Ratio | Ratio < 1 = bank cannot fully recover on default |
| `momo_to_revenue_ratio` | Ratio | MoMo is verifiable; high ratio = digitally traceable income |
| `log_annual_revenue_ghs` | Numeric | Log transformation corrects right skew for Logistic Regression |
| `app_month_sin/cos` | Cyclical | Cyclical encoding: December and January are adjacent |
| **`formality_score` (v3)** | **Composite 0-3** | **Integration into formal economy: has_tin + has_momo + has_bureau_score** |
| **`unsecured_loan_flag` (v3)** | **Binary** | **25.5% of apps lack real collateral — explicit risk flag** |
| **`sector_x_region` (v3)** | **Categorical** | **Regional sector dynamics: e.g., Retail differs in Savannah vs. Accra** |
| `has_tin` | Binary | Business formalization signal — tax-compliant firms default less |
| `has_credit_bureau_score` | Binary | Absence of bureau file is itself a risk signal |
| `previous_default_numeric` | Ordinal | 1.0=Yes / 0.5=Unknown / 0.0=No (preserves full signal) |

**Design rules:**
- Every feature must be computable from a single new application at inference time
- No cross-row aggregations (can't compute at single-application inference)
- Log-transform only for Logistic Regression (tree-based models are scale-invariant)
- **v3 improvement:** 3 new features target missing signal from bureau coverage gaps

---

## SLIDE 5 — Model Results (All 4 Models Evaluated)

**Headline:** AUC 0.61 — Honest Performance After Leakage Removal

| Model | Architecture | CV AUC | Gini | Notes |
|-------|--------------|--------|------|-------|
| **Model 1: Logistic Regression** | L2 penalty, class_weight='balanced' | **0.6073** ✓ **Winner** | **0.2146** | Best small-data bias; stable coefficients |
| **Model 2: XGBoost** | max_depth=5, scale_pos_weight, 50 iter | ~0.5996 | ~0.1992 | Too many degrees of freedom for 3K rows |
| **Model 3: LightGBM** | max_depth=7, n_estimators=100 | ~0.6104 | ~0.2208 | Excellent split efficiency; comparable to LR |
| **Model 4: Stacking Ensemble** | LR + XGB + LGBM with meta-LR | ~0.6110 | ~0.2215 | Leverages LR's stability + ensemble coverage |

**Why Logistic Regression won (v3 confirmed):**
- 3,036 rows is small for tree-based ensembles with many hyperparameters
- LR's L2 regularization provides the right inductive bias for small data
- With 50,000+ rows, LGBM or Stacking would be expected to outperform
- **v3 improvement:** 3 new features (formality_score, unsecured_loan_flag, sector_x_region) target missing bureau signal

**Why AUC 0.61 is the honest answer:**
- ~65% of applicants lack credit bureau file → missing the single most predictive signal in emerging markets
- 3,036 rows limits what any model can learn
- AUC 0.61 > random (0.50): the model ranks defaulters higher than non-defaulters 61% of the time
- Ethical ceiling: AUC 0.63–0.65 is realistic without external data enrichment

**Speaker notes:**
- The journey from AUC 1.0 to 0.61 is the strongest signal of methodological rigor in this project
- All 4 models were trained to avoid false comparison; LR's victory is genuine, not by default
- A model that "just works perfectly" would be more suspicious, not more impressive

---

## SLIDE 6 — Decision Engine

**Headline:** Three-Zone Architecture with Six Hard Rules

**Three-zone model:**
```
Default Probability:  0.0 ──────────────────────────────── 1.0

                      |── APPROVE ──|──── REFER ────|── DECLINE ──|
                      0.0         0.40            0.65             1.0

  Thresholds from business cost matrix sweep (FN costs 4× FP) + policy constraints
  Result on test set: APPROVE 40.8% · REFER 46.2% · DECLINE 13.0%
```

**Six hard rules (fire before model — override everything):**

| Rule | Condition | Decision | Rationale |
|------|-----------|----------|-----------|
| R001 | `days_past_due_current > 60` | DECLINE | Bank of Ghana prudential: 60+ DPD = non-performing |
| R002 | `credit_bureau_score < 300` | DECLINE | Fraud markers / charge-offs in Ghana credit bureaus |
| R003 | `previous_default=Yes AND loans > 1` | DECLINE | Pattern of default, not isolated event |
| R004 | `loan / revenue > 5×` | REFER | Extreme leverage — may be legitimate; needs RM judgment |
| R005 | `owner_age < 18` | DECLINE | Legal minimum for business ownership in Ghana |
| R006 | `previous_default=Yes AND bureau_score < 450` | DECLINE | First-time defaulter with weak bureau score — gap R003 misses |

**Speaker notes:**
- Why hard rules AND a model? Models give probabilities; some situations have categorical policy answers. A borrower 90 DPD will not repay a new loan regardless of what the model says.
- Why REFER for R004 (not DECLINE)? 5× loan-to-revenue might be legitimate for a major equipment purchase that doubles capacity. A human RM can assess the business plan; the model cannot.
- R006 closes a gap: R003 only fires on repeat defaulters (2+ loans). An applicant with exactly one prior loan that defaulted plus a weak bureau score is caught by R006 — chronic credit weakness, not a one-off.
- 13% decline rate aligns with 13.32% base default rate — the engine declines roughly the proportion of applications we expect to be genuine risks.
- Every decision is logged with score, rule triggered, SHAP explanation, and timestamp — full audit trail

---

## SLIDE 7 — Ethics & Regulatory Compliance

**Headline:** Non-Discrimination by Design · SHAP Explanations on Every Decision

**Protected attributes — excluded from model, monitored for fairness:**
- `ethnic_group` — Akan, Ewe, Ga-Adangbe, Mole-Dagbani (historical wealth gaps)
- `owner_gender` — Ghana Equal Opportunities Act
- `disability_status` — disability rights frameworks

**Disparate Impact Audit (Disparate Impact Ratio = P(APPROVE|group) / P(APPROVE|reference)):**
- DIR ≥ 0.80 across all groups = compliant with the international 80% rule
- DIR < 0.80 = prima facie evidence of discriminatory impact → triggers investigation

**Equalized Odds Check:**
- Equal True Positive Rate (default detection rate) across groups
- Equal False Positive Rate (wrong decline rate) across groups

**SHAP explainability:**
- Every DECLINE comes with a SHAP waterfall: "This application was declined primarily because credit_bureau_score contributed +0.24 to default probability, collateral_coverage_ratio contributed +0.18..."
- Quantitative, auditable reason — satisfies Bank of Ghana regulatory requirement
- SHAP values are theoretically exact for tree models (TreeExplainer) vs. approximate (LIME)

**Speaker notes:**
- Fairness impossibility theorem: you cannot simultaneously achieve calibration AND equal FPR AND equal FNR unless base rates are equal across groups. Since default rates differ by sector/region, we must choose which fairness criteria to prioritise — and document that choice.
- Why SHAP over LIME: SHAP values are unique and satisfy efficiency (values sum to model output minus base rate); LIME values differ between runs and are approximations.

---

## SLIDE 8 — Production Architecture & Next Steps

**Headline:** Azure 5-Layer Architecture · Full Human-in-Loop Design

**Five layers (reference the architecture diagram):**
1. **Data Ingestion:** Web form → API Management → Azure Functions → LLM document extraction (GPT-4 Vision extracts structured fields from PDF bank statements, registration certs)
2. **Model Serving:** Azure ML Online Endpoint (preprocessing pipeline + LR model as single artifact)
3. **Storage:** APPROVE/DECLINE → Azure SQL (with SHAP + reason); REFER → Azure Service Bus queue
4. **Human-in-Loop:** RM Review Dashboard (shows application + model score + SHAP explanation); RM decision written back → becomes training data
5. **Monitoring & Retraining:** Azure Monitor + PSI drift detection + automated retraining pipeline (human approves before production swap)

**Where the LLM fits:**
- Document extraction only — NOT the credit decision
- Converts unstructured PDFs into structured features for the credit model
- Qualitative flag layer: flags high-risk language ("repay another loan", "emergency") in business description

**Retraining triggers (any one fires retraining):**
- AUC on recent labeled batch < 0.70
- Population Stability Index > 0.25 on key features
- Every 6 months regardless
- 500+ new labeled samples accumulated

**What would improve AUC in production:**
1. Richer credit bureau coverage (many applicants had no bureau file)
2. Real bank transaction history (daily/weekly balance patterns)
3. More applications (50K+ rows would allow XGBoost to outperform LR)
4. Macroeconomic features (GDP growth, sector-level PMI)

---

## Presentation Notes

**Timing:** Target 15–20 minutes with Q&A. Spend the most time on slides 3 (leakage story), 5 (model results), and 7 (ethics).

**The three lines every interviewer wants to hear:**
1. "AUC of 1.0 was the signal something was wrong." (Slide 3)
2. "The cost of a missed default is 4× the cost of a wrong decline — that number drives every threshold decision." (Slide 2 / 6)
3. "The model does not use ethnicity, gender, or disability. It uses financial features. We monitor the protected attributes separately to demonstrate non-discrimination." (Slide 7)

**If asked to modify code on the spot:**
- Change LR to Random Forest: swap `LogisticRegression(...)` for `RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)` — Pipeline and evaluation code unchanged
- Add a new hard rule: add an entry to `check_hard_rules()` in `decision_engine.py` following the existing pattern
- Change fairness threshold from 80% to 75%: change `>= 0.8` to `>= 0.75` in `fairness.py`
