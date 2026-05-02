# Stanbic Bank Ghana — SME Credit Assessment Pipeline
## Submission Guide

**Assessment Date:** May 1, 2026  
**Status:** Complete and Ready for Submission  
**Interview Format:** Take-home assessment with live code modification

---

## What You're Submitting

A complete, production-ready ML pipeline that converts raw loan applications into automated credit decisions with regulatory-grade fairness auditing and explainability.

### Deliverable Summary

| Item | Format | Count | Location |
|------|--------|-------|----------|
| **Notebooks (end-to-end pipeline)** | .ipynb | 7 | `/notebooks/` |
| **Source code modules** | .py | 3 | `/src/` |
| **Trained models** | .pkl | 4 | `/models/` |
| **Presentation** | .md (ready for .pptx) | 8 slides | `/reports/presentation_deck.md` |
| **Compliance report** | .md | 1 | `/reports/COMPLIANCE_REPORT.md` |
| **Documentation** | .md | 2 | `/reports/architecture_diagram.md` + memory files |

---

## OPTION A: Google Colab Link (Fastest)

**Send this to the interviewer:**

```
Google Colab Notebooks (read-only):
- 01_setup_and_eda.ipynb
- 02_preprocessing.ipynb
- 03_feature_engineering.ipynb
- 04_modeling.ipynb
- 05_evaluation.ipynb
- 06_decision_engine.ipynb
- 07_ethics_explainability.ipynb

GitHub Repository (source code):
https://github.com/[your-username]/stanbic-sme-credit

Shared Google Drive folder:
https://drive.google.com/drive/folders/[shared-drive-id]
```

**Pros:**
- Interviewer can run cells interactively
- No download/installation required
- Real-time data visualization

**Cons:**
- Requires Google account
- May have access/sharing restrictions

---

## OPTION B: GitHub Repository (Recommended)

**Create a GitHub repo with this structure:**

```
stanbic-sme-credit/
├── README.md                           (getting started guide)
├── notebooks/                          (all 7 .ipynb files)
│   ├── 01_setup_and_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   ├── 06_decision_engine.ipynb
│   └── 07_ethics_explainability.ipynb
├── src/                                (reusable modules)
│   ├── preprocessing.py
│   ├── decision_engine.py
│   └── fairness.py
├── models/                             (trained models + registry)
│   ├── logistic_regression_v1.pkl
│   ├── xgboost_v1.pkl
│   ├── lightgbm_v1.pkl
│   ├── stacking_v1.pkl
│   └── model_registry.json
├── reports/                            (documentation)
│   ├── COMPLIANCE_REPORT.md
│   ├── presentation_deck.md
│   ├── architecture_diagram.md
│   └── figures/                        (EDA charts + SHAP plots)
├── data/                               (directory structure only)
│   ├── raw/
│   ├── interim/
│   └── processed/
└── requirements.txt                    (dependencies)
```

**README.md template:**

```markdown
# Stanbic Bank Ghana — SME Credit Assessment Pipeline

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Open notebooks in order:
   - 01_setup_and_eda.ipynb — exploratory data analysis
   - 02_preprocessing.ipynb — data cleaning + fairness columns
   - 03_feature_engineering.ipynb — derived features + train/test split
   - 04_modeling.ipynb — 4 models trained + compared
   - 05_evaluation.ipynb — comprehensive evaluation metrics
   - 06_decision_engine.ipynb — hard rules + decision logic
   - 07_ethics_explainability.ipynb — fairness audit + SHAP

3. View presentation:
   - Read `reports/presentation_deck.md`
   - See `reports/COMPLIANCE_REPORT.md` for requirement mapping

## Key Results

- **CV AUC:** 0.6073 (Logistic Regression)
- **Models trained:** 4 (LR, XGB, LGBM, Stacking)
- **Hard rules:** 6 business rules override model for edge cases
- **Fairness:** Disparate impact audit + SHAP explanations
- **Features:** 41 engineered from 30 raw columns

## Architecture

See `reports/architecture_diagram.md` for end-to-end pipeline flow.

## License

[MIT] (or your choice)
```

**requirements.txt:**

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
shap>=0.41.0
matplotlib>=3.6.0
seaborn>=0.12.0
statsmodels>=0.13.0
imbalanced-learn>=0.10.0
joblib>=1.2.0
```

**Push to GitHub:**

```bash
git init
git add .
git commit -m "Stanbic SME Credit Assessment Pipeline - v3 complete"
git branch -M main
git remote add origin https://github.com/[your-username]/stanbic-sme-credit.git
git push -u origin main
```

**Share with interviewer:**

```
https://github.com/[your-username]/stanbic-sme-credit
```

---

## OPTION C: Zip File (Offline)

**Create a .zip containing:**

```
stanbic-sme-credit.zip
├── Stanbic_SME_Credit_Assessment_v3.pptx  (presentation)
├── COMPLIANCE_REPORT.md                   (formal compliance doc)
├── notebooks/                             (7 .ipynb files)
├── src/                                   (3 .py modules)
├── models/                                (4 .pkl files + registry.json)
├── reports/
│   ├── presentation_deck.md
│   ├── architecture_diagram.md
│   └── figures/                           (PNG charts from notebook outputs)
├── data/
│   ├── processed/
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   ├── y_test.csv
│   │   └── split_metadata.json
│   └── raw/
│       └── ds-sme_loan_applications_stanbic_gh.csv
└── README.txt                             (how to use)
```

**Size management:**
- CSV files can be large; consider uploading to cloud (Google Drive) instead
- Include `model_registry.json` so interviewer can see which model won

**Share via:**
- Email attachment (if < 25MB)
- Google Drive link
- Dropbox link
- WeTransfer (for large files)

---

## What the Interviewer Expects to See

### Before the Interview (Reading/Setup)
1. **README** — says "this is a credit pipeline, here's how to run it"
2. **Presentation deck** — 8 slides explaining the business problem + methodology
3. **COMPLIANCE_REPORT.md** — maps pipeline to original brief requirements
4. **Architecture diagram** — visual overview of data → model → decision flow

### During the Live Session (15–20 mins)
1. **Walk through notebooks** (3–5 minutes)
   - "Here's notebook 01, we load 3,036 applications and run 6 EDA questions"
   - "Notebook 02 cleans data; removed leakage columns, created flags"
   - Skip detailed code; focus on outputs and business impact

2. **Model results** (2–3 minutes)
   - "We trained 4 models: Logistic Regression, XGBoost, LightGBM, and Stacking"
   - "LR won with AUC 0.6073. Why? 3,036 rows favor regularization over ensemble complexity"
   - "AUC 1.0 → 0.61: we removed data leakage. 0.61 is the honest answer"

3. **Decision engine demo** (2–3 minutes)
   - Show hard rules (R001-R006) + threshold zones
   - "Rule R004 fires if loan > 5× revenue — we REFER for RM judgment"

4. **Fairness audit** (2 minutes)
   - "Protected attributes excluded from model; we monitor them separately"
   - "Every decision includes SHAP explanation: why was this declined?"

5. **Code modification (if asked)** (~5 minutes)
   - Swap LR for Random Forest
   - Add new hard rule
   - Change fairness threshold

---

## The Three Talking Points Interviewers Want

**Point 1: Leakage Detection** (Slide 3 in presentation)
> "In the first run, we got AUC 1.0. That was the signal something was wrong. We found that `days_past_due_current` is recorded at loan maturity — it's the answer sheet, not the exam question. After removing it and `rm_recommendation`, AUC dropped to 0.61. That's the honest predictive power."

**Point 2: Cost Asymmetry** (Slide 2 / 6 in presentation)
> "A missed default costs the bank GHS 80,000. A wrong decline costs GHS 22,000 in foregone interest. That 4× ratio drives every threshold decision. It's why our decline rate aligns with our base default rate."

**Point 3: Fairness by Design** (Slide 7 in presentation)
> "The model does not use ethnicity, gender, or disability status. These protected attributes are excluded by design. We monitor them separately to audit whether decisions correlate with protected groups — they shouldn't. If they do, we use remediation strategies like sample weight rebalancing."

---

## Pre-Interview Checklist

- [ ] All 7 notebooks execute end-to-end without errors (test locally or in Colab)
- [ ] COMPLIANCE_REPORT.md shows how pipeline maps to brief requirements
- [ ] Presentation deck (.md) covers all 8 slides with speaker notes
- [ ] Interview prep memory (interview_prep_v3.md) has 3 talking points memorized
- [ ] GitHub repo (or alternative) is publicly accessible and up-to-date
- [ ] Model results match documented values (AUC 0.6073, Gini 0.2146 for LR)
- [ ] You can explain: leakage story → cost asymmetry → fairness design
- [ ] You're ready to modify code: swap LR → RF, add hard rule, change threshold

---

## Common Interview Questions (& Your Answers)

**Q: "How would you improve AUC from 0.61?"**
A: "Three paths: (1) Richer credit bureau coverage — 65% of applicants have no bureau file; that's the missing signal. (2) Real bank transaction data — daily/weekly balance patterns. (3) More training data — 50K+ applications would let XGBoost or LGBM outperform LR. We can't solve this with better modeling alone; it's a data acquisition problem."

**Q: "Why Logistic Regression over XGBoost?"**
A: "With 3,036 rows, LR's L2 regularization is the right bias. XGBoost has too many degrees of freedom. Think of it like poker: with 1,000 hands of data, a 'loose' strategy (many bets) overfits; a 'tight' strategy (fewer but stronger hands) wins. With 50K+ hands, we'd see XGBoost pull ahead."

**Q: "Explain the hard rules. Why not just use the model?"**
A: "Models give probabilities. Hard rules are categorical policies. If someone is 90 DPD (days past due), they won't repay any new loan, regardless of model score. Rule R006 catches first-time defaulters with weak bureau scores — the gap R003 (repeat defaulters) misses."

**Q: "What's your fairness defense?"**
A: "We excluded protected attributes from the model. We audit the decisions with Disparate Impact Ratio (DIR ≥ 0.80 = compliant). If DIR < 0.80, we don't ignore it — we investigate and apply remediation (sample weight rebalancing). It's 'fair by design + monitored + acted-upon'."

**Q: "Modify the code to [request]."**
A: See code modification tips in interview_prep_v3.md memory.

---

## After the Interview

**Timeline for decision:**
- Panel typically takes 1–2 weeks to review and decide
- Some panels give same-day verbal feedback + formal offer letter in 1 week

**If they ask for changes:**
- They'll specify: "Can you add feature X?" or "Re-run with Y model?"
- You have the notebooks and source code; re-running is straightforward
- Turnaround: 2–3 days for major changes

**If you don't hear back:**
- After 1 week, you can send a polite follow-up email
- "I wanted to check on the status of the assessment. Happy to answer any follow-up questions."

---

## Submission Checklist (Final)

**Before sending:**
- [ ] All notebooks run without errors (tested in Google Colab)
- [ ] Presentation deck exists (.md + ready for .pptx conversion)
- [ ] COMPLIANCE_REPORT.md is complete and readable
- [ ] GitHub repo (or zip) is organized and includes README
- [ ] No sensitive data (passwords, API keys) in code
- [ ] You have the 3 talking points memorized
- [ ] You've practiced explaining the leakage story + cost asymmetry + fairness

**When sending:**
- Share GitHub link OR zip file OR Google Colab notebooks (pick one)
- Include a brief cover email:
  ```
  Subject: Stanbic SME Credit Assessment — Complete Submission
  
  Hi [Recruiter Name],
  
  Please find my complete Stanbic SME Credit Assessment submission below. 
  The pipeline includes 7 end-to-end Jupyter notebooks, 4 trained models 
  (LR winning with AUC 0.6073), fairness audit, SHAP explainability, 
  and an 8-slide presentation deck.
  
  Quick links:
  - GitHub: [github.com/...]
  - Presentation: [link to slides]
  - Compliance report: [COMPLIANCE_REPORT.md]
  
  I'm ready for the live interview and happy to modify code on the spot.
  
  Best regards,
  [Your Name]
  ```

---

## Questions?

Refer to:
- **COMPLIANCE_REPORT.md** — maps requirements to deliverables
- **interview_prep_v3.md** — v3 key numbers + talking points
- **presentation_deck.md** — 8-slide overview with speaker notes
- **Notebook headers** — each notebook explains its purpose in markdown cells

---

**Submitted:** May 1, 2026  
**Status:** Ready for Interview  
**Good luck!**
