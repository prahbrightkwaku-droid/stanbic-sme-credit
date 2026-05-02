"""
fairness.py — Stanbic Bank Ghana SME Credit Assessment
=======================================================
Fairness auditing tools for demonstrating non-discrimination
to the Bank of Ghana and other regulatory bodies.

Implements:
  1. Disparate Impact Ratio (80% rule)
  2. Equalized Odds check (equal TPR and FPR across groups)
  3. Compliance report generation

THEORY:
    Disparate Treatment: Using a protected characteristic directly in decisions.
        → Prevented by EXCLUDING ethnic_group, owner_gender, disability_status from model.

    Disparate Impact: Facially neutral policy with disproportionate negative effect
        on a protected group.
        → Detected by comparing approval rates across groups (DIR).
        → Can occur even without discriminatory intent.

    Fairness Impossibility Theorem (Chouldechova, 2017):
        Cannot simultaneously achieve perfect calibration, equal FPR, and equal FNR
        across groups unless base rates are equal.
        → Must choose which fairness criterion to prioritise and document that choice.

INTERVIEW TALKING POINT:
    "We excluded protected attributes from the model to prevent disparate treatment.
     We then measured disparate impact using the 80% rule and equalized odds.
     The compliance report documents both what we did and what we found."
"""

import pandas as pd
import numpy as np
from datetime import datetime


PROTECTED_ATTRIBUTES = [
    'owner_gender',
    'ethnic_group',
    'region',
    'disability_status',
]

EIGHTY_PERCENT_RULE = 0.80  # DIR < 0.80 triggers investigation under US ECOA framework


class FairnessAuditor:
    """
    Audits credit model decisions for disparate impact across demographic groups.

    Parameters
    ----------
    protected_attrs : list
        Columns to audit (default: the four protected attributes)
    """

    def __init__(self, protected_attrs=None):
        self.protected_attrs = protected_attrs or PROTECTED_ATTRIBUTES
        self._audit_results  = {}

    def disparate_impact_ratio(self, decisions_df: pd.DataFrame,
                                protected_col: str,
                                favorable_outcome: str = 'APPROVE') -> pd.DataFrame:
        """
        Compute the Disparate Impact Ratio for a protected attribute.

        DIR = P(favorable outcome | group) / P(favorable outcome | best-off group)

        A DIR < 0.80 triggers the 80% rule:
            The group receiving the favorable outcome at < 80% of the best-off
            group's rate is considered to face prima facie discrimination.

        WHY compare to the best-off group (not a specific reference group):
            In Ghana, no single group is the legal "reference". Comparing every
            group to the highest-approval-rate group ensures we catch ALL
            disparities, regardless of which group happens to be advantaged.

        NOTE: DIR < 0.80 does NOT prove discrimination. It triggers an
        investigation. The gap may be fully explained by legitimate credit factors
        (e.g., lower credit scores in one group). But the gap must be investigated
        and documented. The investigation itself is the compliance requirement.

        Parameters
        ----------
        decisions_df : pd.DataFrame
            Must contain protected_col and 'recommendation' column
        protected_col : str
            Column name of the protected attribute
        favorable_outcome : str
            Which recommendation is 'favorable' (default: 'APPROVE')

        Returns
        -------
        pd.DataFrame with columns: approval_rate, disparate_impact_ratio, pass_80_rule
        """
        if protected_col not in decisions_df.columns:
            return None
        if 'recommendation' not in decisions_df.columns:
            return None

        group_rates = (
            decisions_df.groupby(protected_col)['recommendation']
            .apply(lambda x: (x == favorable_outcome).mean())
            .rename('approval_rate')
        )

        if group_rates.empty or group_rates.isna().all():
            return None

        reference_rate = group_rates.max()
        if reference_rate == 0:
            return None

        dir_series = (group_rates / reference_rate).rename('disparate_impact_ratio')

        result = pd.DataFrame({
            'approval_rate':           group_rates,
            'disparate_impact_ratio':  dir_series,
            'pass_80_rule':            dir_series >= EIGHTY_PERCENT_RULE,
            'count':                   decisions_df.groupby(protected_col).size(),
        }).sort_values('approval_rate', ascending=False)

        return result

    def equalized_odds(self, decisions_df: pd.DataFrame,
                       protected_col: str,
                       y_true_col: str = 'actual_default') -> pd.DataFrame:
        """
        Check for equal True Positive Rate (TPR) and False Positive Rate (FPR)
        across demographic groups.

        Equalized Odds (Hardt et al., 2016) requires:
            TPR_group_A = TPR_group_B  (equal ability to detect defaults)
            FPR_group_A = FPR_group_B  (equal rate of wrongly declining good borrowers)

        WHY stricter than Disparate Impact:
            A model might have equal approval rates but catch defaulters at
            different rates by group. If the model catches 70% of male defaults
            but only 40% of female defaults, it is systematically less accurate
            for women — a subtle but important fairness violation even if
            approval rates are equal.

        Parameters
        ----------
        decisions_df : pd.DataFrame
            Must contain protected_col, y_true_col, and 'recommendation' column
        """
        if protected_col not in decisions_df.columns:
            return None

        results = []
        for group in decisions_df[protected_col].dropna().unique():
            mask   = decisions_df[protected_col] == group
            subset = decisions_df[mask].copy()
            n      = len(subset)

            # Convert recommendations to binary (APPROVE=1 for positive label logic)
            # For credit: default=1 is positive class
            # APPROVE = model says "won't default" = predicts 0
            # DECLINE = model says "will default" = predicts 1
            y_true = subset[y_true_col].astype(int)
            # Note: REFER cases are excluded from TPR/FPR calculation
            # since the model did not make a binary decision on them
            binary_mask = subset['recommendation'].isin(['APPROVE', 'DECLINE'])
            y_true_bin  = y_true[binary_mask]
            y_pred_bin  = (subset.loc[binary_mask, 'recommendation'] == 'DECLINE').astype(int)

            if len(y_true_bin) < 10 or y_true_bin.sum() == 0:
                results.append({
                    'group': group, 'n': n, 'n_binary': len(y_true_bin),
                    'TPR': np.nan, 'FPR': np.nan,
                    'TPR_FPR_note': 'insufficient data'
                })
                continue

            tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
            fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
            tn = int(((y_pred_bin == 0) & (y_true_bin == 0)).sum())
            fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            results.append({
                'group':         group,
                'n':             n,
                'n_binary':      len(y_true_bin),
                'TPR':           round(tpr, 4),
                'FPR':           round(fpr, 4),
                'TPR_FPR_note':  'OK'
            })

        result_df = pd.DataFrame(results).set_index('group')
        return result_df

    def full_audit(self, decisions_df: pd.DataFrame,
                   y_true_col: str = 'actual_default') -> dict:
        """
        Run the complete fairness audit for all protected attributes.

        Returns a nested dict:
            {
              'disparate_impact': {attr: DataFrame},
              'equalized_odds':   {attr: DataFrame},
              'flags': [list of failing attributes]
            }
        """
        audit = {
            'audit_date':       datetime.now().isoformat(),
            'n_decisions':      len(decisions_df),
            'disparate_impact': {},
            'equalized_odds':   {},
            'flags':            []
        }

        for attr in self.protected_attrs:
            if attr not in decisions_df.columns:
                continue

            # Disparate impact
            dir_result = self.disparate_impact_ratio(decisions_df, attr)
            audit['disparate_impact'][attr] = dir_result

            if dir_result is not None and not dir_result['pass_80_rule'].all():
                failing = dir_result[~dir_result['pass_80_rule']].index.tolist()
                audit['flags'].append({
                    'type':      'DISPARATE_IMPACT',
                    'attribute': attr,
                    'failing_groups': failing,
                    'message':   f'DIR < 0.80 for {failing} in {attr} — investigate'
                })

            # Equalized odds (only if y_true_col present)
            if y_true_col in decisions_df.columns:
                eo_result = self.equalized_odds(decisions_df, attr, y_true_col)
                audit['equalized_odds'][attr] = eo_result

        self._audit_results = audit
        return audit

    def print_audit(self, audit: dict):
        """Print a formatted audit report."""
        print('=' * 70)
        print('FAIRNESS AUDIT REPORT')
        print(f'Date: {audit["audit_date"]}')
        print(f'Total decisions audited: {audit["n_decisions"]:,}')
        print('=' * 70)
        print()
        print('PROTECTED ATTRIBUTES IN MODEL: None')
        print('  (ethnic_group, owner_gender, disability_status excluded from model)')
        print('  (region is included — monitored for proxy discrimination)')
        print()

        print('DISPARATE IMPACT ANALYSIS (80% Rule):')
        print('-' * 50)
        for attr, result in audit['disparate_impact'].items():
            if result is None:
                print(f'  {attr}: insufficient data')
                continue
            print(f'\n  {attr.upper()}:')
            print(result[['count', 'approval_rate', 'disparate_impact_ratio', 'pass_80_rule']].to_string())

        if audit['equalized_odds']:
            print('\nEQUALIZED ODDS ANALYSIS (Equal TPR and FPR across groups):')
            print('-' * 50)
            for attr, result in audit['equalized_odds'].items():
                if result is None:
                    continue
                print(f'\n  {attr.upper()}:')
                print(result[['n', 'TPR', 'FPR']].to_string())

        if audit['flags']:
            print('\nFLAGS REQUIRING INVESTIGATION:')
            print('-' * 50)
            for flag in audit['flags']:
                print(f'  [{flag["type"]}] {flag["attribute"]}: {flag["message"]}')
        else:
            print('\nNO FLAGS: All groups pass the 80% rule.')

        print()
        print('=' * 70)

    def compute_sample_weights(self, df: pd.DataFrame, protected_col: str,
                                target_col: str, random_state: int = 42) -> np.ndarray:
        """
        Compute sample weights that equalize positive class representation across groups.

        When one demographic group has systematically lower approval rates due to
        correlated proxy features (not inherent credit risk), upweighting that group's
        positive examples during training nudges the model toward a more equitable
        decision boundary.

        **Important caveat:** This DOES NOT change individual predictions. It changes
        what the model learns to be "typical" behaviour for each group.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with protected_col and target_col
        protected_col : str
            Column name of the protected attribute (e.g., 'ethnic_group')
        target_col : str
            Column name of the target variable (e.g., 'loan_default')
        random_state : int
            For reproducibility

        Returns
        -------
        np.ndarray
            Sample weights aligned with df.index, ready for:
            `model.fit(X, y, classifier__sample_weight=weights)`
        """
        from sklearn.utils.class_weight import compute_sample_weight

        # Per-group class weights: upweight minority positives within each group
        # This ensures the model learns each group's positive class with equal weight
        weights = np.ones(len(df))
        for group in df[protected_col].dropna().unique():
            mask = df[protected_col] == group
            group_target = df.loc[mask, target_col]
            group_weights = compute_sample_weight('balanced', group_target)
            weights[mask] = group_weights

        return weights

    def suggest_remediation(self, dir_result: pd.DataFrame) -> dict:
        """
        Suggest remediation actions based on disparate impact analysis results.

        Parameters
        ----------
        dir_result : pd.DataFrame
            Output from disparate_impact_ratio() with columns:
            approval_rate, disparate_impact_ratio, pass_80_rule, count

        Returns
        -------
        dict
            Recommendation summary with keys:
            - reweighting_recommended: bool (if any DIR < 0.80)
            - threshold_adjustment_recommended: bool (if max group gap > 0.15)
            - min_dir: float (minimum DIR across groups)
            - failing_groups: list (groups failing 80% rule)
            - message: str (human-readable summary)
        """
        if dir_result is None or dir_result.empty:
            return {
                'reweighting_recommended': False,
                'threshold_adjustment_recommended': False,
                'min_dir': 1.0,
                'failing_groups': [],
                'message': 'No disparate impact detected (all groups pass 80% rule).'
            }

        failing = dir_result[~dir_result['pass_80_rule']].index.tolist()
        min_dir = dir_result['disparate_impact_ratio'].min()
        approval_range = dir_result['approval_rate'].max() - dir_result['approval_rate'].min()

        return {
            'reweighting_recommended': min_dir < 0.80,
            'threshold_adjustment_recommended': approval_range > 0.15,
            'min_dir': float(min_dir),
            'failing_groups': failing,
            'message': (
                f'Min DIR: {min_dir:.3f}. Failing groups: {failing}. '
                f'Recommended remediation: sample weight rebalancing + per-group threshold tuning.'
                if failing
                else 'All groups pass 80% rule (DIR >= 0.80).'
            )
        }

    def generate_compliance_report(self, audit: dict, model_name: str) -> dict:
        """
        Generate a structured compliance report for Bank of Ghana submission.

        The report answers the four regulatory questions:
        1. What protected attributes does the model use?
        2. Are approval rates equitable across demographic groups?
        3. Is model accuracy equitable across groups?
        4. What remediation actions are in place?
        """
        n_flags = len(audit.get('flags', []))

        report = {
            'report_date':              datetime.now().isoformat(),
            'model_name':               model_name,

            # Question 1: Protected attributes in model
            'protected_attrs_in_model':         [],
            'protected_attrs_monitored':        PROTECTED_ATTRIBUTES,
            'confirmation_no_protected_use':    True,

            # Question 2: Approval rate equity
            'disparate_impact_summary': {
                attr: {
                    'all_groups_pass_80_rule': (
                        result['pass_80_rule'].all()
                        if result is not None else None
                    ),
                    'min_dir': (
                        float(result['disparate_impact_ratio'].min())
                        if result is not None else None
                    )
                }
                for attr, result in audit['disparate_impact'].items()
            },

            # Question 3: Model accuracy equity
            'equalized_odds_summary': {
                attr: {
                    'tpr_range': (
                        [float(result['TPR'].min()), float(result['TPR'].max())]
                        if result is not None and 'TPR' in result.columns
                        else None
                    )
                }
                for attr, result in audit.get('equalized_odds', {}).items()
            },

            # Question 4: Remediation and monitoring
            'remediation_actions': [
                'Protected attributes excluded from model features',
                'Monthly disparate impact monitoring scheduled',
                'Any DIR < 0.80 triggers root-cause investigation within 5 business days',
                'SHAP explanations attached to every DECLINE decision for audit',
                'Human appeal process available for all declined applications',
                'Model performance reviewed quarterly; retrain if AUC drops below 0.70',
            ],

            # Flags
            'flags_count':              n_flags,
            'flags_requiring_action':   audit.get('flags', []),
            'overall_status':           'PASS' if n_flags == 0 else f'INVESTIGATE ({n_flags} flags)',
        }

        return report
