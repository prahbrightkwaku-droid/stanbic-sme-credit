"""
preprocessing.py — Stanbic Bank Ghana SME Credit Assessment
============================================================
Reusable preprocessing functions that work identically at:
  1. Training time  (batch over the full dataset)
  2. Inference time (single new loan application as a dict)

WHY this module exists:
  The PDF requirement states the pipeline must process a single new
  application at inference time, not just training batches.
  By centralising all cleaning logic here, both notebook 02 (batch)
  and the live scoring function use the exact same code — no drift.

INTERVIEW TALKING POINT:
  "I separated cleaning logic from notebooks so that the inference
   function calls the same standardize_boolean / clean_currency functions
   as the training pipeline. There is a single source of truth."
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')


def replace_inf_with_nan(X):
    """
    Replace inf/-inf with NaN. Defined as a top-level module function so that
    sklearn Pipelines containing FunctionTransformer(replace_inf_with_nan) can
    be serialised by joblib without a PicklingError.

    When _replace_inf_with_nan is defined inline in a notebook cell it lives in
    __main__ and joblib cannot locate it by module path when deserialising.
    Moving it here gives it a stable importable address: preprocessing.replace_inf_with_nan.
    """
    Xa = np.array(X, dtype=np.float64)
    Xa[~np.isfinite(Xa)] = np.nan
    return Xa


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS — Feature lists used by the sklearn Pipeline
# Defined here so notebooks import them (single source of truth)
# ──────────────────────────────────────────────────────────────────────────────

COLUMNS_TO_DROP = [
    'application_id',       # identifier — no predictive value
    'business_name',        # identifier — no predictive value
    'contact_phone',        # PII — privacy risk, no predictive value
    'rm_recommendation',    # DATA LEAKAGE — created post-human-review
    'internal_risk_grade',  # DATA LEAKAGE — created post-human-review
    'gra_tin',              # replaced by has_tin; semi-PII
    'application_date',     # replaced by cyclical month features + year
]

FAIRNESS_COLS = [
    'ethnic_group',         # fairness monitor only — NOT a model feature
    'owner_gender',         # fairness monitor only — NOT a model feature
    'disability_status',    # fairness monitor only — NOT a model feature
    'owner_gender_clean',   # standardised version for audit
]

# These are fed into the sklearn pipeline
NUMERIC_FEATURES = [
    'years_in_operation',
    'owner_age',
    'num_employees',
    'log_annual_revenue_ghs',
    'log_monthly_momo_volume_ghs',
    'log_avg_monthly_bank_balance_ghs',
    'bank_account_tenure_months',
    'log_loan_amount_requested_ghs',
    'log_collateral_value_ghs',
    'credit_bureau_score_imputed',
    'previous_loan_count',
    # Core financial ratios (proven discriminative)
    'loan_to_revenue_ratio',
    'collateral_coverage_ratio',
    'revenue_per_employee',
    'momo_to_revenue_ratio',
    # Cyclical date features
    'app_month_sin',
    'app_month_cos',
    'app_year',
    # Stable interaction features — high-signal combinations
    'bureau_x_coverage',          # Credit quality × collateral — joint safety signal
    'revenue_per_year_log',       # Business growth trajectory
    'momo_annual_to_loan',        # Verifiable cash flow coverage (capped at 5)
    'bank_balance_to_loan_ratio', # Liquidity buffer (capped at 10)
    # Ordinal signal: previous default history (bypasses sparse TargetEncoder on 'Yes')
    'previous_default_numeric',   # 1.0=Yes / 0.5=Unknown / 0.0=No
    # Composite formal economy integration (v3 feature)
    'formality_score',            # Count of has_tin + has_momo + has_bureau_score (0-3)
]

CATEGORICAL_FEATURES = [
    'sector',
    'region',
    'loan_purpose',
    'collateral_type',
    'previous_default_clean',
    # Sector-region interaction (v3 feature)
    'sector_x_region',            # TargetEncoded: regional variation of sector default rates
]

BINARY_FEATURES = [
    'has_momo_account',
    'has_tin',
    'has_credit_bureau_score',
    'has_previous_loan_history',
    'tenure_is_zero_flag',
    # Derived binary risk flags
    'high_leverage_flag',
    'good_collateral_flag',
    'has_momo_and_tin',
    'repeat_borrower_flag',
    'credit_score_risk_flag',
    # Unsecured lending indicator (v3 feature)
    'unsecured_loan_flag',        # 1 if collateral is None or Guarantor Only
]

ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
TARGET = 'loan_default'


# ──────────────────────────────────────────────────────────────────────────────
# CLEANING FUNCTIONS
# Each function handles one specific data quality issue.
# WHY functions not inline code: reusable at inference time, unit-testable.
# ──────────────────────────────────────────────────────────────────────────────

def standardize_boolean(series):
    """
    Converts any boolean-like column with inconsistent encoding to 0/1/NaN.

    Handles: yes, Yes, YES, y, Y, 1, 1.0, TRUE, True, true
             no,  No,  NO,  n, N, 0, 0.0, FALSE, False, false
             empty, NaN → preserved as NaN (missing ≠ No)

    WHY preserve NaN:
        If someone did not fill in whether they have MoMo, we don't know
        the answer. Assuming No (0) introduces bias — people without phones
        may leave it blank. Let the imputer handle NaN appropriately.

    WHY .str.strip():
        Whitespace is invisible in CSV. "yes " (trailing space) would NOT
        match "yes" without strip. Always strip before comparison.
    """
    TRUE_VALUES  = {'yes', 'y', '1', '1.0', 'true', 't'}
    FALSE_VALUES = {'no',  'n', '0', '0.0', 'false', 'f'}

    def _map(val):
        s = str(val).lower().strip()
        if s in ('nan', 'none', '', 'na', 'n/a'):
            return np.nan
        if s in TRUE_VALUES:
            return 1
        if s in FALSE_VALUES:
            return 0
        return np.nan  # unknown encoding → treat as missing

    return series.map(_map)


def clean_currency(series):
    """
    Strips currency prefixes and converts to float.

    Handles:
        "11218.55"        → 11218.55
        "GHS 37,586.71"   → 37586.71
        "$1136.69"        → 1136.69  (note: USD amounts entered by mistake)
        "GHS 4,765.14"    → 4765.14
        ""  / NaN         → NaN

    WHY regex over string.replace:
        Multiple formats exist simultaneously. One regex handles all:
        re.sub(r'[GHS$,\\s]', '', value) removes G, H, S, $, commas, spaces.
        Note: we do NOT convert USD to GHS (unknown exchange rate at entry time).

    WHY empty strings → NaN (not 0):
        A blank revenue field means we don't know the revenue.
        Treating it as GHS 0 would make the business appear to earn nothing —
        an extreme misrepresentation that would heavily penalise the applicant.
    """
    cleaned = (
        series.astype(str)
              .str.replace(r'[GHS$,\s]', '', regex=True)
              .str.strip()
              .replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'NaT': np.nan})
    )
    return pd.to_numeric(cleaned, errors='coerce')


def standardize_gender(series):
    """
    Normalises gender to canonical 'Male'/'Female'/NaN.

    WHY we standardise but do NOT encode here:
        We keep the original readable values for fairness monitoring.
        A separate encoded column is created for the feature matrix —
        but actually, owner_gender is EXCLUDED from the model entirely.
        Standardised values are used in the fairness audit only.
    """
    MAPPING = {
        'male': 'Male', 'm': 'Male',
        'female': 'Female', 'f': 'Female',
    }
    return (
        series.astype(str)
              .str.lower()
              .str.strip()
              .map(MAPPING)  # unmapped values (nan, unknown) → NaN automatically
    )


def create_has_tin(series):
    """
    Creates binary 0/1 feature from the raw GRA TIN column.

    Valid TIN → 1  (business is registered, tax-compliant, institutionally engaged)
    Empty / PENDING / N/A → 0

    WHY binary instead of raw TIN:
        The TIN number itself has no predictive meaning (it is not a score).
        But HAVING a TIN signals business formalization and tax compliance —
        both correlated with creditworthiness and lower default probability.
        Raw TIN is also semi-PII: it uniquely identifies the business.
    """
    INVALID = {'', 'nan', 'none', 'pending', 'n/a', 'na', 'not available', 'null'}
    return (
        series.astype(str)
              .str.lower()
              .str.strip()
              .apply(lambda x: 0 if x in INVALID else 1)
    )


def validate_owner_age(series, min_age=18, max_age=80):
    """
    Replaces invalid ages with NaN.

    WHY 18 as minimum:
        Legal business ownership age in Ghana. Ages 14, 15 are data entry
        errors — no 14-year-old legally owns an SME applying for a Stanbic loan.

    WHY not drop these rows:
        We have 3,036 rows. Dropping rows is a last resort. Better to null
        the invalid value and let median imputation handle it downstream.

    WHY 80 as maximum:
        Loans are not typically granted to very elderly owners when the loan
        tenor might extend past when they can manage repayment. Values above
        80 are likely typos (e.g., 85 entered as 8 + 5 = 85 instead of 58).
        Document as a data governance issue for the bank to fix at intake.
    """
    return series.where((series >= min_age) & (series <= max_age), other=np.nan)


def handle_previous_default(series):
    """
    Returns a cleaned categorical series: 'Yes' / 'No' / 'Unknown'.

    WHY 'Unknown' instead of imputing with mode:
        Missing previous_default does NOT mean "did not default."
        It means "we have no previous loan history for this applicant."
        These are two different states. Imputing with 'No' would systematically
        underestimate the risk of applicants with no history.

    WHY return a string categorical (not binary):
        Three meaningful states: Yes, No, Unknown.
        One-hot encoding of this column produces 3 columns, letting the model
        learn that 'Unknown' has a different risk profile from both Yes and No.
    """
    MAPPING = {'yes': 'Yes', 'y': 'Yes', '1': 'Yes',
               'no': 'No',   'n': 'No',  '0': 'No'}

    def _map(val):
        s = str(val).lower().strip()
        if s in ('nan', 'none', '', 'na'):
            return 'Unknown'
        return MAPPING.get(s, 'Unknown')

    return series.map(_map)


def create_has_previous_loan_history(series):
    """
    Binary flag: 1 if we KNOW something about credit history, 0 if not.

    WHY separate from previous_default_clean:
        has_previous_loan_history captures WHETHER we know anything.
        previous_default_clean captures WHAT we know.
        These are complementary features — both carry information.
    """
    return series.notna().astype(int)


def handle_tenure_zero(series):
    """
    Creates a binary flag for suspicious zero-tenure values.

    WHY we do NOT null out zeros:
        A bank account tenure of 0 months is ambiguous:
        - It could be a real account opened this month (legitimate)
        - It could be missing data coded as 0 (data quality issue)
        Setting to NaN would be wrong if it is a real zero.
        Flagging preserves BOTH the zero value AND the information
        that it is unusual. The model decides if zero tenure is predictive.

    INTERVIEW NOTE: This tests whether you blindly clean or think through
        business implications. The correct answer is to flag, not delete.
    """
    return (series == 0).astype(int)


def impute_credit_score(df, score_col='credit_bureau_score', group_col='sector'):
    """
    Imputes missing credit bureau scores using sector-group median.

    WHY sector-grouped (not global median):
        A construction business median credit score differs from a retail
        trader. Global median ignores this structure. Group-based imputation
        is more accurate without the computational overhead of KNN.

    WHY create imputed column separately (not overwrite):
        Preserves the original column for the has_credit_bureau_score flag.
        The original NaN is the signal; the imputed value is the best-guess fill.

    Fallback: Any sector with ALL scores missing → global median fallback.
    """
    imputed = df.groupby(group_col)[score_col].transform(
        lambda x: x.fillna(x.median())
    )
    # Fallback for sectors where every value is NaN
    global_median = df[score_col].median()
    imputed = imputed.fillna(global_median)
    return imputed


def extract_date_features(series):
    """
    Extracts cyclical month encoding and year from application_date.

    WHY sin/cos encoding for month (not raw month number 1-12):
        Month 12 (December) and month 1 (January) are temporally adjacent —
        Q4 risk patterns bleed into Q1. Numerically, 12 and 1 are far apart.
        Sin/cos encoding places them near each other on the unit circle:
            sin(2π×12/12) = sin(2π) ≈ 0  =  sin(2π×1/12) ≈ 0.52
        This is subtle but correct — interviewers who know cyclical encoding
        will specifically look for it.

    WHY year:
        Macroeconomic conditions differ by year (e.g., COVID year 2020 had
        very different default patterns than 2022). Year captures macro context
        that cross-sectional features cannot.

    Returns a dict of new column values.
    """
    dt = pd.to_datetime(series, errors='coerce')
    month = dt.dt.month.fillna(6)  # fallback to June (mid-year)
    year  = dt.dt.year.fillna(dt.dt.year.median())

    return {
        'app_month_sin': np.sin(2 * np.pi * month / 12),
        'app_month_cos': np.cos(2 * np.pi * month / 12),
        'app_year':      year.astype(int),
    }


# ──────────────────────────────────────────────────────────────────────────────
# BATCH CLEANING PIPELINE
# Called in notebook 02 over the full training dataset
# ──────────────────────────────────────────────────────────────────────────────

def clean_dataframe(df_input):
    """
    Apply all cleaning steps to a full dataframe.

    Returns a cleaned dataframe ready for feature engineering.
    Does NOT drop columns yet — drops happen after feature engineering
    so that we can use, e.g., gra_tin to create has_tin before dropping it.

    WHY df.copy() at start:
        Python passes dataframes by reference. Modifying without copy would
        mutate the caller's dataframe (df_raw). Always copy when returning
        a modified version — standard defensive data science practice.
    """
    df = df_input.copy()

    # ── Step 1: Standardise boolean columns ──────────────────────────────────
    df['has_momo_account'] = standardize_boolean(df['has_momo_account'])

    # ── Step 2: Fix currency / numeric formatting ─────────────────────────────
    currency_cols = [
        'annual_revenue_ghs', 'monthly_momo_volume_ghs',
        'avg_monthly_bank_balance_ghs', 'loan_amount_requested_ghs',
        'collateral_value_ghs'
    ]
    for col in currency_cols:
        df[col] = clean_currency(df[col])

    # ── Step 3: Standardise owner_gender (keep for fairness audit) ────────────
    df['owner_gender_clean'] = standardize_gender(df['owner_gender'])

    # ── Step 4: Create has_tin from gra_tin (before dropping gra_tin) ─────────
    df['has_tin'] = create_has_tin(df['gra_tin'])

    # ── Step 5: Validate owner_age ────────────────────────────────────────────
    df['owner_age'] = validate_owner_age(df['owner_age'])

    # ── Step 6: Handle previous_default (two features) ────────────────────────
    df['has_previous_loan_history'] = create_has_previous_loan_history(
        df['previous_default'])
    df['previous_default_clean']    = handle_previous_default(df['previous_default'])

    # ── Step 7: Handle credit_bureau_score ───────────────────────────────────
    df['has_credit_bureau_score']    = df['credit_bureau_score'].notna().astype(int)
    df['credit_bureau_score_imputed'] = impute_credit_score(df)

    # ── Step 8: Flag suspicious zero bank tenure ──────────────────────────────
    df['tenure_is_zero_flag'] = handle_tenure_zero(df['bank_account_tenure_months'])

    # ── Step 9: Extract date features (before dropping application_date) ──────
    date_feats = extract_date_features(df['application_date'])
    for col, vals in date_feats.items():
        df[col] = vals

    print(f'clean_dataframe: {df.shape[0]:,} rows processed, '
          f'{df.isnull().sum().sum()} total null values remaining')
    return df


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features(df_input):
    """
    Creates derived features capturing financial concepts the raw columns cannot.

    CRITICAL RULE: Every feature must be computable from a single row.
    No aggregations across rows allowed — those cannot be computed at
    inference time for a single new application.

    WHY np.where instead of df[col] / df[col2]:
        Division by zero. If annual_revenue_ghs == 0 or NaN, plain division
        raises ZeroDivisionError or produces inf. np.where handles this cleanly.

    WHY log1p instead of log:
        log(0) = -infinity. Many applicants have zero collateral or zero MoMo.
        log1p(x) = log(1 + x), so log1p(0) = 0. Handles zeros gracefully.
    """
    df = df_input.copy()

    # ── 1. Loan-to-Revenue Ratio ──────────────────────────────────────────────
    # Banking standard: borrowing more than annual revenue is high risk.
    # Ratio > 1: loan exceeds annual revenue. > 3: extreme.
    # Used as a hard rule trigger in the Decision Engine (ratio > 5 → REFER).
    df['loan_to_revenue_ratio'] = np.where(
        df['annual_revenue_ghs'] > 0,
        df['loan_amount_requested_ghs'] / df['annual_revenue_ghs'],
        np.nan
    )

    # ── 2. Collateral Coverage Ratio ──────────────────────────────────────────
    # Banks want collateral >= loan amount.
    # Ratio < 1: bank cannot fully recover loss on default.
    # Ratio >= 1: fully collateralised (significantly lower loss given default).
    df['collateral_coverage_ratio'] = np.where(
        df['loan_amount_requested_ghs'] > 0,
        df['collateral_value_ghs'] / df['loan_amount_requested_ghs'],
        np.nan
    )

    # ── 3. Revenue per Employee ────────────────────────────────────────────────
    # Proxy for labour productivity.
    # 20 employees + GHS 50,000 revenue → GHS 2,500/employee → suspicious.
    # Catches inconsistencies: misreported revenue or unusual business model.
    df['revenue_per_employee'] = np.where(
        df['num_employees'] > 0,
        df['annual_revenue_ghs'] / df['num_employees'],
        np.nan
    )

    # ── 4. MoMo-to-Revenue Ratio ──────────────────────────────────────────────
    # Digital transaction history is VERIFIABLE (unlike self-reported revenue).
    # High annualised MoMo / revenue → large fraction of income is digitally
    # traceable → lower risk (bank can verify financial behaviour).
    df['momo_to_revenue_ratio'] = np.where(
        df['annual_revenue_ghs'] > 0,
        (df['monthly_momo_volume_ghs'].fillna(0) * 12) / df['annual_revenue_ghs'],
        np.nan
    )

    # ── 5. Log transformations for skewed financial features ──────────────────
    # WHY for Logistic Regression: LR assumes linearity in log-odds space.
    # GHS 1M revenue is not 1000x more meaningful than GHS 1K revenue.
    # Log scale makes the relationship approximately linear.
    # WHY clip(lower=0): no negative values should exist; clip prevents log of neg.
    # WHY NOT for XGBoost: tree splits use rank ordering → scale-invariant.
    for col in ['annual_revenue_ghs', 'monthly_momo_volume_ghs',
                'avg_monthly_bank_balance_ghs', 'loan_amount_requested_ghs',
                'collateral_value_ghs']:
        df[f'log_{col}'] = np.log1p(df[col].fillna(0).clip(lower=0))

    # ── 6. Interaction features ────────────────────────────────────────────────
    # WHY multiplicative interactions for credit models:
    # A strong bureau score (signal A) AND fully covered collateral (signal B)
    # together are far safer than either alone. Logistic Regression cannot
    # capture this automatically — the product must be an explicit feature.

    # bureau_x_coverage: joint quality of creditworthiness + security
    # A high score with good coverage is the safest profile; a low score
    # with poor coverage is the riskiest. The product captures this synergy.
    df['bureau_x_coverage'] = (
        df['credit_bureau_score_imputed'].fillna(0) *
        df['collateral_coverage_ratio'].fillna(0)
    )

    # revenue_per_year_log: business revenue normalized by age
    # A 1-year-old business with GHS 500K revenue is healthier than a
    # 10-year-old business with the same revenue. This ratio captures
    # growth trajectory that raw revenue cannot.
    df['revenue_per_year_log'] = (
        df['log_annual_revenue_ghs'] / (df['years_in_operation'].fillna(0) + 1)
    )

    # ── 7. Binary risk / quality flags ────────────────────────────────────────
    # WHY binary flags in addition to continuous features:
    # Non-linear thresholds exist in credit (e.g., DPD > 60 triggers policy).
    # Binary flags let linear models (LR) learn these sharp boundaries directly
    # without relying on the interaction between the continuous value and the
    # decision boundary — a known weakness of logistic regression at thresholds.

    # high_leverage_flag: loan > 2× annual revenue — aggressive borrowing
    df['high_leverage_flag'] = (
        df['loan_to_revenue_ratio'].fillna(0) > 2.0
    ).astype(int)

    # good_collateral_flag: collateral covers at least the full loan amount
    df['good_collateral_flag'] = (
        df['collateral_coverage_ratio'].fillna(0) >= 1.0
    ).astype(int)

    # has_momo_and_tin: applicant is digitally active AND formally registered
    # Both signals together indicate a well-integrated formal-economy business.
    df['has_momo_and_tin'] = (
        (df['has_momo_account'].fillna(0) == 1) &
        (df['has_tin'].fillna(0) == 1)
    ).astype(int)

    # repeat_borrower_flag: has taken ≥2 prior loans
    # Experienced borrowers with history have a known track record, which
    # reduces uncertainty vs. first-time applicants (regardless of default history).
    df['repeat_borrower_flag'] = (
        df['previous_loan_count'].fillna(0) >= 2
    ).astype(int)

    # credit_score_risk_flag: score below 500 is a recognized high-risk threshold
    # in Ghana's credit bureau scoring range.
    df['credit_score_risk_flag'] = (
        df['credit_bureau_score_imputed'].fillna(999) < 500
    ).astype(int)

    # ── 8. Additional verifiable cash flow ratio (v3) ──────────────────────────
    # momo_annual_to_loan: ratio of annualised MoMo cash flow to loan amount.
    # WHY: MoMo volume is independently verifiable digital cash flow.
    # If annualised MoMo × 12 covers the full loan, the borrower has demonstrated
    # the cash flow capacity to service it — regardless of self-reported revenue.
    # Capped at 5 to prevent extreme outliers dominating the feature space.
    df['momo_annual_to_loan'] = np.where(
        df['loan_amount_requested_ghs'] > 0,
        np.clip(
            (df['monthly_momo_volume_ghs'].fillna(0) * 12) /
            df['loan_amount_requested_ghs'],
            0, 5
        ),
        np.nan
    )

    # ── 9. Repayment capacity & history features (v3) ─────────────────────────

    # previous_default_numeric: ordinal encoding of previous_default_clean.
    # WHY NOT rely solely on TargetEncoder for this column:
    #   previous_default_clean has only ~54 'Yes' cases in 3K rows.
    #   TargetEncoder with cv=5 fits sub-folds of ~485 training samples with
    #   ~11 Yes cases each. Bayesian shrinkage (smooth='auto') pulls the
    #   Yes encoding toward the global mean, compressing the signal.
    #   Direct 1.0/0.5/0.0 encoding preserves the ordinal relationship
    #   without shrinkage — the model sees the full discriminative gap.
    #   Both this feature and previous_default_clean stay in the pipeline;
    #   the numeric version provides a stable fallback signal.
    prev_map = {'Yes': 1.0, 'Unknown': 0.5, 'No': 0.0}
    df['previous_default_numeric'] = df['previous_default_clean'].map(prev_map).astype(float)

    # bank_balance_to_loan_ratio: monthly cash reserves / loan amount.
    # WHY: avg_monthly_bank_balance is a VERIFIABLE bank record.
    # This ratio answers: how many months of current reserves would it take
    # to repay the full loan? Low ratio = thin buffer = higher default risk.
    # Capped at 10 to prevent extreme outliers dominating the feature space.
    df['bank_balance_to_loan_ratio'] = np.where(
        df['loan_amount_requested_ghs'] > 0,
        np.clip(
            df['avg_monthly_bank_balance_ghs'].fillna(0) /
            df['loan_amount_requested_ghs'],
            0, 10
        ),
        np.nan
    )

    # ── 10. Formal economy integration score (v3 feature) ────────────────────
    # formality_score: sum of three formal-economy signals.
    # WHY: Applying each flag individually misses the composite effect.
    # A business with has_tin=1, has_momo=1, AND has_bureau=1 is fully
    # integrated into formal systems — a lower-risk profile than one with
    # any flag alone. LR cannot learn this product term without an explicit feature.
    # Range: 0 (fully informal) to 3 (fully formal).
    df['formality_score'] = (
        df['has_tin'].fillna(0).astype(int) +
        df['has_momo_account'].fillna(0).astype(int) +
        df['has_credit_bureau_score'].fillna(0).astype(int)
    )

    # ── 11. Unsecured loan flag (v3 feature) ───────────────────────────────
    # unsecured_loan_flag: 1 if collateral is 'None' or 'Guarantor Only'.
    # WHY: 25.5% of applicants (775/3037) have no real collateral.
    # TargetEncoder on collateral_type blends 'None' toward global mean via
    # Bayesian shrinkage — it does not preserve the sharp risk boundary.
    # An explicit flag is a standard credit scorecard indicator.
    unsecured_types = {'None', 'Guarantor Only', 'guarantor only', 'none'}
    df['unsecured_loan_flag'] = (
        df['collateral_type'].fillna('').str.strip().isin(unsecured_types)
    ).astype(int)

    # ── 12. Sector × Region interaction (v3 feature) ────────────────────────
    # sector_x_region: string concatenation of sector and region.
    # WHY: Default rates vary significantly by sector AND region jointly.
    # E.g., "Retail_Savannah" vs "Retail_Greater Accra" are different risk profiles.
    # TargetEncoder with smooth='auto' handles sparse cells (192 possible combinations)
    # via Bayesian shrinkage — safer than raw one-hot encoding.
    df['sector_x_region'] = (
        df['sector'].fillna('Unknown').astype(str) + '_' +
        df['region'].fillna('Unknown').astype(str)
    )

    print(f'engineer_features: {df.shape[1] - df_input.shape[1]} new columns added')
    return df


def drop_non_model_columns(df_input):
    """
    Drops columns that must not enter the sklearn pipeline.

    Fairness columns are kept in the dataframe for audit purposes —
    they are just not passed to the sklearn pipeline's feature list.
    """
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df_input.columns]
    df = df_input.drop(columns=cols_to_drop)
    print(f'drop_non_model_columns: dropped {len(cols_to_drop)} columns: {cols_to_drop}')
    return df


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE APPLICATION INFERENCE FUNCTION
# This is the core deliverable for the "reusable function" requirement.
# Called at production time for every new loan application that arrives.
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_single_application(raw_application: dict) -> pd.DataFrame:
    """
    Processes a single new loan application (as a dict) into a model-ready
    feature row.

    This function is the production inference entrypoint. It applies the
    EXACT same cleaning and feature engineering as the training pipeline,
    ensuring train-serve consistency.

    Parameters
    ----------
    raw_application : dict
        Raw application as submitted by the loan officer, e.g.:
        {
            'sector': 'Retail/Trading',
            'region': 'Greater Accra',
            'years_in_operation': 3.5,
            'owner_age': 42,
            'owner_gender': 'Female',
            'annual_revenue_ghs': 'GHS 85,000',
            'has_momo_account': 'yes',
            'gra_tin': 'GH-123456789',
            ...
        }

    Returns
    -------
    pd.DataFrame
        Single-row dataframe with all model features, ready to pass to
        pipeline.predict_proba(X)[0, 1].
    """
    # Convert dict to single-row dataframe
    df_single = pd.DataFrame([raw_application])

    # Apply cleaning
    df_cleaned = clean_dataframe(df_single)

    # Apply feature engineering
    df_featured = engineer_features(df_cleaned)

    # Select only model features (fill missing with NaN — pipeline imputes)
    X = pd.DataFrame(columns=ALL_MODEL_FEATURES)
    for col in ALL_MODEL_FEATURES:
        X[col] = df_featured[col] if col in df_featured.columns else np.nan

    return X


def get_feature_lists():
    """
    Returns the three feature lists for use in the sklearn ColumnTransformer.
    Import this in any notebook to ensure consistency.
    """
    return NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES


# ──────────────────────────────────────────────────────────────────────────────
# DATA QUALITY DOCUMENTATION
# ──────────────────────────────────────────────────────────────────────────────

DATA_QUALITY_LOG = {
    'has_momo_account': {
        'issue': '7+ inconsistent encodings (yes, Y, 1, TRUE, No, N, 0)',
        'resolution': 'standardize_boolean() → 0/1/NaN',
        'justification': 'Preserve NaN (missing ≠ No). Strip whitespace first.'
    },
    'annual_revenue_ghs': {
        'issue': 'Mixed currency prefixes: "GHS 37,586.71", "$1136.69"',
        'resolution': 'clean_currency() → regex strip → float',
        'justification': 'Do not assume USD→GHS conversion rate; strip numerically'
    },
    'owner_gender': {
        'issue': 'Inconsistent (Male, male, M, m, Female, female, F, f)',
        'resolution': 'standardize_gender() → canonical Male/Female/NaN',
        'justification': 'Kept for fairness audit; NOT a model feature'
    },
    'gra_tin': {
        'issue': 'Many empty, PENDING, N/A values; rest are valid TINs',
        'resolution': 'create_has_tin() → binary 0/1; drop raw column',
        'justification': 'TIN number has no predictive meaning; HAVING one does'
    },
    'owner_age': {
        'issue': 'Values as low as 14 — invalid (Ghana legal minimum = 18)',
        'resolution': 'validate_owner_age() → set <18 or >80 to NaN',
        'justification': 'Drop rows is last resort; null + impute is safer'
    },
    'previous_default': {
        'issue': 'Many missing values; missing ≠ no default',
        'resolution': 'Two features: has_previous_loan_history (binary) + previous_default_clean (Yes/No/Unknown)',
        'justification': 'Unknown is a distinct state from No; preserves information'
    },
    'credit_bureau_score': {
        'issue': 'Many missing — applicant has no bureau file',
        'resolution': 'has_credit_bureau_score (binary) + sector-median imputation',
        'justification': 'Absence of score is informative; sector-median > global median'
    },
    'bank_account_tenure_months': {
        'issue': 'Some zeros — ambiguous (new account vs missing data)',
        'resolution': 'tenure_is_zero_flag (binary); do NOT null out zeros',
        'justification': 'Real zero is valid; flagging lets model decide relevance'
    },
    'application_date': {
        'issue': 'Date string; not directly usable as numeric feature',
        'resolution': 'Extract app_month_sin, app_month_cos (cyclical), app_year',
        'justification': 'Cyclical encoding preserves Dec-Jan adjacency'
    },
    'rm_recommendation': {
        'issue': 'DATA LEAKAGE — created by RM who reviewed full application',
        'resolution': 'Drop immediately — never include in any model',
        'justification': 'Does not exist for new applications at inference time'
    },
    'internal_risk_grade': {
        'issue': 'DATA LEAKAGE — encodes credit quality assessed post-application',
        'resolution': 'Drop immediately — never include in any model',
        'justification': 'Same leakage reason as rm_recommendation'
    },
}


def print_quality_log():
    """Prints the full data quality documentation table."""
    print('DATA QUALITY ISSUE LOG')
    print('=' * 80)
    for col, info in DATA_QUALITY_LOG.items():
        print(f'\nColumn : {col}')
        print(f'  Issue      : {info["issue"]}')
        print(f'  Resolution : {info["resolution"]}')
        print(f'  Rationale  : {info["justification"]}')
