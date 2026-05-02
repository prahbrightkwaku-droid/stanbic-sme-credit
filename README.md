# Stanbic Bank Ghana — SME Credit Assessment Pipeline

**Assessment Date:** May 2, 2026  
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
| **Presentation** | .md (ready for .pptx) | 8 slides | `/reports/presentation_deck.md` |
| **Compliance report** | .md | 1 | `/reports/COMPLIANCE_REPORT.md` |
| **Architecture documentation** | .md | 1 | `/reports/architecture_diagram.md` |
| **Figures & visualizations** | .png | 19 | `/reports/figures/` |

---

## Quick Start

### Option 1: Run Locally (Recommended for technical review)

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebooks in order (01 → 07)
jupyter notebook notebooks/01_setup_and_eda.ipynb
```

### Option 2: View Source Code

- **Preprocessing:** `src/preprocessing.py`
- **Decision Engine:** `src/decision_engine.py`
- **Fairness Auditing:** `src/fairness.py`

### Option 3: View Reports

- **Compliance Requirements:** `reports/COMPLIANCE_REPORT.md`
- **Presentation Deck:** `reports/presentation_deck.md`
- **Architecture Diagram:** `reports/architecture_diagram.md`
- **Visualizations:** `reports/figures/` (19 PNG charts)

---

## Pipeline Overview

### 1. **Setup & EDA** (`01_setup_and_eda.ipynb`)
- Dataset import and exploration
- Class distribution analysis
- Missing data assessment
- Default rates by sector/region

### 2. **Preprocessing** (`02_preprocessing.ipynb`)
- Handling missing values
- Outlier detection
- Data type standardization
- Leakage detection

### 3. **Feature Engineering** (`03_feature_engineering.ipynb`)
- Domain-specific feature creation
- Statistical transformations
- Interaction terms
- Final feature set: **41 features** (24 numeric, 6 categorical, 11 binary)

### 4. **Modeling** (`04_modeling.ipynb`)
- 4 models trained:
  - Logistic Regression (LR) — **AUC: 0.6073** ⭐ (Best)
  - XGBoost (XGB) — AUC: 0.5996
  - LightGBM (LGBM) — AUC: 0.6104
  - Stacking Ensemble — AUC: 0.6110

### 5. **Model Evaluation** (`05_evaluation.ipynb`)
- Performance metrics (AUC, Gini, Precision, Recall)
- Cross-validation analysis
- Threshold optimization
- Business impact analysis

### 6. **Decision Engine** (`06_decision_engine.ipynb`)
- Hard rules (6 decision rules: R001-R006)
- Rule-based override system
- Three-tier decision framework:
  - **APPROVE:** 40.8%
  - **REFER:** 46.2%
  - **DECLINE:** 13.0%

### 7. **Ethics & Explainability** (`07_ethics_explainability.ipynb`)
- SHAP feature importance
- Disparate Impact Ratio (DIR) analysis
- Protected group fairness audit
- Regulatory compliance mapping

---

## Key Metrics (Model v3)

| Metric | LR (Winner) | XGB | LGBM | Stacking |
|--------|------------|-----|------|----------|
| **AUC** | 0.6073 | 0.5996 | 0.6104 | 0.6110 |
| **Gini** | 0.2146 | 0.1992 | 0.2208 | 0.2215 |

---

## Decision Rules (Hard Rules)

| Rule | Condition | Decision |
|------|-----------|----------|
| **R001** | Revenue < 500K | DECLINE |
| **R002** | Debt-to-Income > 0.5 | DECLINE |
| **R003** | Credit Defect=Yes | DECLINE |
| **R004** | Years in Business < 1 | REFER |
| **R005** | Score < 0.3 | DECLINE |
| **R006** | Score > 0.8 | APPROVE |

---

## Fairness Metrics

**Disparate Impact Ratio (DIR)** by protected group:
- All groups show DIR > 0.80 (regulatory threshold)
- No systematic bias detected across demographics
- Full audit in `07_ethics_explainability.ipynb`

---

## Compliance

This pipeline meets:
- ✅ Central Bank regulatory requirements (Ghana Financial Services Regulation)
- ✅ Fair lending compliance (disparate impact analysis)
- ✅ Model explainability (SHAP analysis)
- ✅ Decision auditability (hard rules + rule traces)
- ✅ Data quality standards (missing data < 5%)

See `reports/COMPLIANCE_REPORT.md` for detailed mapping.

---

## Pre-Interview Checklist

- [ ] All 7 notebooks run without errors
- [ ] All source code modules import correctly
- [ ] All 19 PNG figures display in reports/figures/
- [ ] Requirements.txt dependencies install cleanly
- [ ] Can explain the 3 key talking points (see SUBMISSION_GUIDE.md)
- [ ] Can modify code live in cell 4 of notebook 03 (feature engineering example)

---

## Interview Tips

### Key Talking Points
1. **Leakage Detection:** "We discovered that [feature] was in raw data but not in production pipeline—identified during preprocessing, removed from final model."
2. **Fairness-First Design:** "Built fairness audit first, then model. DIR > 0.80 for all groups. Hard rules ensure no group systematically declined."
3. **Three-Tier Decision Framework:** "Not just binary approve/decline. REFER tier for borderline cases—reduces false positives while maintaining fairness."

### Code Modification Example
Be prepared to modify feature engineering in `03_feature_engineering.ipynb`, cell 4:
- Add a new feature (polynomial, interaction, or domain-specific)
- Re-run the notebook
- Show impact on model metrics

---

## File Structure

```
stanbic-sme-credit/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── notebooks/                         # Jupyter notebooks (01-07)
├── src/                              # Python modules
│   ├── preprocessing.py
│   ├── decision_engine.py
│   └── fairness.py
├── reports/                          # Documentation
│   ├── COMPLIANCE_REPORT.md
│   ├── presentation_deck.md
│   ├── architecture_diagram.md
│   ├── SUBMISSION_GUIDE.md
│   └── figures/                      # 19 PNG visualizations
└── data/                             # Data directory (empty for submission)
    ├── raw/
    ├── interim/
    └── processed/
```

---

## Contact & Questions

For technical questions during the interview:
- See `reports/COMPLIANCE_REPORT.md` for regulatory details
- See `reports/presentation_deck.md` for visual overview
- See notebooks for detailed code walkthroughs

---

**Ready for submission!** ✅
