"""
decision_engine.py — Stanbic Bank Ghana SME Credit Assessment
=============================================================
Converts a raw probability score + application dict into a final
business recommendation: APPROVE / DECLINE / REFER TO HUMAN.

Design philosophy:
  1. Hard rules fire FIRST and override the model entirely.
     Some situations have categorical policy answers (e.g., 90 days
     past due) that no probability calculation changes.
  2. Score thresholds handle the probabilistic middle ground.
  3. Every decision is logged with reason, rule, score, and timestamp.

INTERVIEW TALKING POINT:
  "The Decision Engine separates model output from business logic.
   The model gives a probability; the engine converts it to action.
   This means credit policy changes (adjust thresholds, add rules)
   require no model retraining — only engine configuration changes."
"""

import numpy as np
import pandas as pd
from datetime import datetime


class SMECreditDecisionEngine:
    """
    Three-zone decision engine:

    Score < approve_threshold  → APPROVE (model is confident: low risk)
    Score > decline_threshold  → DECLINE (model is confident: high risk)
    Otherwise                  → REFER   (uncertain: human review needed)

    Hard rules are checked first and override the score entirely.

    Parameters
    ----------
    approve_threshold : float
        Applications with default probability below this are auto-approved.
        Default: 0.25 (calibrated via business cost matrix sweep in notebook 06)

    decline_threshold : float
        Applications with default probability above this are auto-declined.
        Default: 0.55 (asymmetric: FN costs more than FP, so decline threshold
        is NOT 1 - approve_threshold)

    WHY class not just functions:
        A class maintains threshold state. Credit committees will adjust thresholds
        over time (e.g., tighten during economic downturns). With a class, you
        change the parameters without touching the decision logic — Open/Closed Principle.
    """

    DEFAULT_APPROVE_THRESHOLD = 0.25
    DEFAULT_DECLINE_THRESHOLD = 0.55

    def __init__(self, approve_threshold=None, decline_threshold=None):
        self.approve_threshold = approve_threshold or self.DEFAULT_APPROVE_THRESHOLD
        self.decline_threshold = decline_threshold or self.DEFAULT_DECLINE_THRESHOLD
        self._hard_rule_violations = []

        assert self.approve_threshold < self.decline_threshold, \
            "approve_threshold must be less than decline_threshold"

    # ──────────────────────────────────────────────────────────────────────────
    # HARD RULES
    # ──────────────────────────────────────────────────────────────────────────

    def check_hard_rules(self, application: dict) -> tuple:
        """
        Check all hard business rules against the raw application.

        Hard rules:
        - Override the model score entirely
        - Represent regulatory requirements or clear policy mandates
        - Are applied BEFORE the score is consulted

        Returns
        -------
        (decision, reason, rule_id) or (None, None, None) if no rule triggered
        """
        triggered = []

        # ── R001: Severe current delinquency ─────────────────────────────────
        # WHY 60 days: Bank of Ghana prudential guidelines consider 60+ DPD
        # as "non-performing" — lending to non-performing borrowers is prohibited.
        # WHY hard rule (not model feature): The model might assign 40% default
        # probability to a borrower with 62 DPD. That is too lenient. This is
        # a categorical disqualifier regardless of other factors.
        dpd = self._safe_float(application.get('days_past_due_current', 0))
        if dpd > 60:
            triggered.append({
                'rule_id': 'R001',
                'rule':    'SEVERE_CURRENT_DELINQUENCY',
                'reason':  f'days_past_due_current={dpd:.0f} exceeds 60-day threshold',
                'decision': 'DECLINE',
                'severity': 1
            })

        # ── R002: Very low credit bureau score ────────────────────────────────
        # WHY 300: Ghana credit bureaus (XDS, CRB Africa, Dun & Bradstreet)
        # use 300-900 range. Below 300 indicates fraud markers, court judgements,
        # or multiple charge-offs — absolute disqualifiers for new lending.
        bureau_score = application.get('credit_bureau_score')
        if bureau_score is not None:
            score_val = self._safe_float(bureau_score)
            if score_val is not None and not np.isnan(score_val) and score_val < 300:
                triggered.append({
                    'rule_id': 'R002',
                    'rule':    'VERY_LOW_BUREAU_SCORE',
                    'reason':  f'credit_bureau_score={score_val:.0f} below minimum 300',
                    'decision': 'DECLINE',
                    'severity': 1
                })

        # ── R003: Repeat defaulter ─────────────────────────────────────────────
        # WHY previous_loan_count > 1: A single past default might be an anomaly
        # (COVID year, illness). Two loans + two defaults = pattern of behaviour.
        # DECLINE on pattern, REFER on isolated incident.
        prev_default = str(application.get('previous_default', '')).lower().strip()
        prev_count   = self._safe_float(application.get('previous_loan_count', 0)) or 0
        if prev_default in ('yes', '1', 'true', 'y') and prev_count > 1:
            triggered.append({
                'rule_id': 'R003',
                'rule':    'REPEAT_DEFAULTER',
                'reason':  f'previous_default=Yes with {prev_count:.0f} prior loans',
                'decision': 'DECLINE',
                'severity': 1
            })

        # ── R004: Extreme loan-to-revenue ratio → REFER (not DECLINE) ─────────
        # WHY REFER not DECLINE: A high ratio might indicate legitimate growth
        # capital needs (e.g., equipment purchase that will double capacity).
        # The model cannot assess a business plan — a human RM can.
        # WHY 5x: Standard banking caps SME lending at 3-4× annual revenue.
        # Above 5× is extreme leverage requiring human judgment.
        revenue  = self._safe_float(application.get('annual_revenue_ghs', 0)) or 0
        loan_amt = self._safe_float(application.get('loan_amount_requested_ghs', 0)) or 0
        if revenue > 0 and loan_amt > 0:
            ltr = loan_amt / revenue
            if ltr > 5:
                triggered.append({
                    'rule_id': 'R004',
                    'rule':    'EXTREME_LOAN_TO_REVENUE',
                    'reason':  f'loan_to_revenue_ratio={ltr:.1f}x exceeds 5x threshold',
                    'decision': 'REFER',
                    'severity': 2  # lower severity — REFER not DECLINE
                })

        # ── R005: Underage owner ──────────────────────────────────────────────
        # WHY: Preprocessing should catch this (set to NaN). But defense-in-depth:
        # if somehow an age < 18 reaches the engine, this is a hard block.
        owner_age = self._safe_float(application.get('owner_age'))
        if owner_age is not None and owner_age < 18:
            triggered.append({
                'rule_id': 'R005',
                'rule':    'UNDERAGE_OWNER',
                'reason':  f'owner_age={owner_age:.0f} below legal minimum (18)',
                'decision': 'DECLINE',
                'severity': 1
            })

        # ── R006: First-time defaulter with weak bureau score ──────────────────
        # WHY not R003 (repeat defaulter): R003 requires previous_loan_count > 1.
        # A single prior default is tolerated when bureau score is healthy — the
        # default may have been a one-off event (illness, COVID). But if the
        # bureau score is also low (< 450), the two signals together indicate
        # chronic credit weakness, not a one-time shock. The model may still score
        # this applicant near the base rate because other features look normal.
        # This rule catches the combination that the model cannot weight correctly
        # due to its limited discrimination (Gini ≈ 0.22).
        # WHY 450 (not 300 like R002): R002 blocks absolute disqualifiers
        # (fraud/charge-offs). 450 is the "caution zone" — below median for
        # legitimate borrowers but above the fraud floor.
        if bureau_score is not None:
            score_val_r6 = self._safe_float(bureau_score)
            if (score_val_r6 is not None and not np.isnan(score_val_r6)
                    and score_val_r6 < 450
                    and prev_default in ('yes', '1', 'true', 'y')):
                triggered.append({
                    'rule_id': 'R006',
                    'rule':    'DEFAULTER_WITH_WEAK_BUREAU',
                    'reason':  (f'previous_default=Yes with credit_bureau_score='
                                f'{score_val_r6:.0f} (below 450 caution threshold)'),
                    'decision': 'DECLINE',
                    'severity': 1
                })

        if not triggered:
            return None, None, None

        # Priority: DECLINE overrides REFER
        # Among multiple triggered rules, take the most severe
        decisions = [r['decision'] for r in triggered]
        final_decision = 'DECLINE' if 'DECLINE' in decisions else 'REFER'
        final_reason   = '; '.join([r['reason'] for r in triggered])
        final_rule_id  = ', '.join([r['rule_id'] for r in triggered])

        self._hard_rule_violations.extend(triggered)
        return final_decision, final_reason, final_rule_id

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN DECISION FUNCTION
    # ──────────────────────────────────────────────────────────────────────────

    def decide(self, probability_score: float, application: dict) -> dict:
        """
        Convert a model probability score into a business recommendation.

        Why return a dict (not just a string):
            In production, every decision must be logged with full context:
            score, reason, rule triggered, timestamp, model version.
            Banking regulations require complete audit trails.
            A plain string loses all this information.

        Parameters
        ----------
        probability_score : float
            Model's predicted probability of default (0 to 1)
        application : dict
            Raw application fields (for hard rule evaluation)

        Returns
        -------
        dict with keys:
            recommendation   : 'APPROVE' / 'DECLINE' / 'REFER'
            probability_score: float
            reason           : human-readable explanation
            hard_rule_triggered: bool
            rule_id          : str or None
            timestamp        : ISO format string
        """
        result = {
            'probability_score':    float(probability_score),
            'recommendation':       None,
            'reason':               None,
            'hard_rule_triggered':  False,
            'rule_id':              None,
            'timestamp':            datetime.now().isoformat(),
        }

        # Step 1: Hard rules (override everything)
        hard_decision, hard_reason, rule_id = self.check_hard_rules(application)
        if hard_decision is not None:
            result['recommendation']      = hard_decision
            result['reason']              = hard_reason
            result['hard_rule_triggered'] = True
            result['rule_id']             = rule_id
            return result

        # Step 2: Score-based thresholds
        score = probability_score
        if score < self.approve_threshold:
            result['recommendation'] = 'APPROVE'
            result['reason'] = (
                f'Score {score:.4f} below approve threshold {self.approve_threshold} — '
                f'model is confident this borrower will repay'
            )
        elif score > self.decline_threshold:
            result['recommendation'] = 'DECLINE'
            result['reason'] = (
                f'Score {score:.4f} above decline threshold {self.decline_threshold} — '
                f'model predicts high default probability'
            )
        else:
            result['recommendation'] = 'REFER'
            result['reason'] = (
                f'Score {score:.4f} in uncertain zone '
                f'[{self.approve_threshold}, {self.decline_threshold}] — '
                f'requires human review'
            )

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # BATCH EVALUATION
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_on_test_set(self, y_prob_array, X_test_raw, y_test):
        """
        Apply the decision engine to the full test set and report results.

        Why this is a separate method:
            We need to show the interviewer/committee that thresholds produce a
            sensible distribution. If 80% of applications go to REFER, the bank
            has not automated anything. Target distribution:
                APPROVE: 40-60%
                DECLINE: 15-25%
                REFER:   20-35%

        Parameters
        ----------
        y_prob_array : array-like, shape (n,)
            Model's predicted probabilities for the test set
        X_test_raw : pd.DataFrame
            Raw (unprocessed) features — needed for hard rule evaluation
        y_test : pd.Series
            True labels — used to validate decision quality
        """
        decisions = []
        for prob, idx in zip(y_prob_array, X_test_raw.index):
            app_dict = X_test_raw.loc[idx].to_dict()
            decision = self.decide(float(prob), app_dict)
            decision['actual_default'] = int(y_test.loc[idx])
            decision['index'] = idx
            decisions.append(decision)

        decisions_df = pd.DataFrame(decisions)

        # Distribution report
        dist = decisions_df['recommendation'].value_counts()
        dist_pct = dist / len(decisions_df) * 100

        print('DECISION ENGINE — Test Set Results')
        print('=' * 50)
        print(f'Total applications evaluated: {len(decisions_df):,}')
        print()
        print('Recommendation Distribution:')
        for rec in ['APPROVE', 'REFER', 'DECLINE']:
            n = dist.get(rec, 0)
            pct = dist_pct.get(rec, 0)
            bar = '█' * int(pct / 2)
            print(f'  {rec:<10} : {n:4d}  ({pct:5.1f}%)  {bar}')

        print()
        print('Quality Validation (actual default rate per recommendation):')
        for rec in ['APPROVE', 'REFER', 'DECLINE']:
            subset = decisions_df[decisions_df['recommendation'] == rec]
            if len(subset) > 0:
                actual_rate = subset['actual_default'].mean()
                print(f'  {rec:<10} : n={len(subset):4d}  actual default rate={actual_rate:.1%}')
                if rec == 'APPROVE':
                    quality = 'GOOD' if actual_rate < 0.10 else 'REVIEW THRESHOLD'
                elif rec == 'DECLINE':
                    quality = 'GOOD' if actual_rate > 0.40 else 'REVIEW THRESHOLD'
                else:
                    quality = 'OK'
                print(f'               Quality: {quality}')

        hard_rule_count = decisions_df['hard_rule_triggered'].sum()
        print(f'\nHard rule overrides: {hard_rule_count} ({hard_rule_count/len(decisions_df):.1%} of decisions)')

        return decisions_df

    # ──────────────────────────────────────────────────────────────────────────
    # THRESHOLD CALIBRATION
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def calibrate_thresholds(y_prob, y_test, fn_cost_multiplier=4.0,
                              min_refer_pct=0.10, max_refer_pct=0.60,
                              min_decline_pct=0.0, max_approve_pct=1.0):
        """
        Find optimal approve/decline thresholds by minimising business cost.

        Business cost model:
            FN (missed default): bank loses principal minus recovery
            FP (declined good):  bank loses interest income
            Ratio: GHS 80K loss / GHS 22K foregone interest ≈ 3.6x → use 4x

        Parameters
        ----------
        y_prob : array-like
            Model predicted probabilities on test set
        y_test : array-like
            True labels
        fn_cost_multiplier : float
            How many times more expensive is a FN than a FP (default 4)
        min_refer_pct : float
            Minimum REFER zone — at least this fraction must go to human review
        max_refer_pct : float
            Maximum REFER zone — automation floor: cap human review queue size.
            Without this, a weak model (AUC ~0.60) correctly minimises cost by
            referring nearly everything, but the pipeline provides no automation.
            Default 0.60 means at least 40% of applications are auto-decided.
        min_decline_pct : float
            Policy floor on the DECLINE zone. Default 0.0 (unconstrained).
            Set to e.g. 0.10 to require the engine auto-declines at least 10%
            of applications — a credit-committee policy constraint, not a model
            parameter. The break-even default rate in the declined pool is
            1 / (1 + fn_cost_multiplier); below that rate, the constraint costs
            money but may be required by credit policy.
        max_approve_pct : float
            Policy ceiling on the APPROVE zone. Default 1.0 (unconstrained).
            Set to e.g. 0.50 to prevent the engine from auto-approving more
            than half of all applications regardless of model confidence.

        Returns
        -------
        (optimal_approve_threshold, optimal_decline_threshold, results_df)
        """
        results = []
        for approve_t in np.arange(0.05, 0.50, 0.05):
            for decline_t in np.arange(approve_t + 0.05, 0.95, 0.05):
                y_decision = np.where(
                    y_prob < approve_t, 'APPROVE',
                    np.where(y_prob > decline_t, 'DECLINE', 'REFER')
                )
                y_arr = np.array(y_test)

                fn = ((y_decision == 'APPROVE') & (y_arr == 1)).sum()
                fp = ((y_decision == 'DECLINE') & (y_arr == 0)).sum()
                refer_pct = (y_decision == 'REFER').mean()
                approve_pct = (y_decision == 'APPROVE').mean()
                decline_pct = (y_decision == 'DECLINE').mean()

                if refer_pct < min_refer_pct or refer_pct > max_refer_pct:
                    continue
                if decline_pct < min_decline_pct:
                    continue  # credit-policy floor on auto-declines
                if approve_pct > max_approve_pct:
                    continue  # credit-policy ceiling on auto-approvals

                declined_mask = y_decision == 'DECLINE'
                declined_default_rate = (
                    y_arr[declined_mask].mean() if declined_mask.sum() > 0 else 0.0
                )

                cost = (fn * fn_cost_multiplier) + fp
                results.append({
                    'approve_threshold':       round(approve_t, 2),
                    'decline_threshold':       round(decline_t, 2),
                    'business_cost':           cost,
                    'fn':                      fn,
                    'fp':                      fp,
                    'refer_pct':               refer_pct,
                    'approve_pct':             approve_pct,
                    'decline_pct':             decline_pct,
                    'declined_default_rate':   declined_default_rate,
                })

        if not results:
            raise ValueError(
                'No threshold combination satisfies all constraints. '
                'Relax min_decline_pct, max_approve_pct, or refer bounds.'
            )

        results_df = pd.DataFrame(results).sort_values('business_cost').reset_index(drop=True)

        best = results_df.iloc[0]
        breakeven = 1.0 / (1.0 + fn_cost_multiplier)

        print('OPTIMAL THRESHOLDS (business cost minimisation):')
        print(f'  approve_threshold : {best["approve_threshold"]}  (APPROVE if score < this)')
        print(f'  decline_threshold : {best["decline_threshold"]}  (DECLINE if score > this)')
        print(f'  business_cost     : {best["business_cost"]:.0f}')
        print(f'  FN (missed defaults) : {best["fn"]:.0f}')
        print(f'  FP (wrongly declined): {best["fp"]:.0f}')
        print(f'  APPROVE rate      : {best["approve_pct"]:.1%}')
        print(f'  REFER rate        : {best["refer_pct"]:.1%}')
        print(f'  DECLINE rate      : {best["decline_pct"]:.1%}')
        print()
        print('DECLINE ZONE QUALITY:')
        actual_dr = best['declined_default_rate']
        verdict = 'COST-POSITIVE' if actual_dr >= breakeven else 'POLICY-COST (below break-even)'
        print(f'  Actual default rate in declined pool : {actual_dr:.1%}')
        print(f'  Break-even default rate (1/(1+{fn_cost_multiplier:.0f})): {breakeven:.1%}')
        print(f'  Assessment : {verdict}')
        if best['decline_pct'] < 0.05:
            print()
            print('  WARNING: decline rate < 5%. Consider setting min_decline_pct=0.10')
            print('  as a credit-policy constraint (independent of model quality).')

        return float(best['approve_threshold']), float(best['decline_threshold']), results_df

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_float(val):
        """Convert val to float, returning None on failure."""
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def get_hard_rules_summary(self):
        """Return dataframe of all hard rule violations from evaluation."""
        return pd.DataFrame(self._hard_rule_violations)

    def get_config(self):
        """Return current threshold configuration as dict."""
        return {
            'approve_threshold': self.approve_threshold,
            'decline_threshold': self.decline_threshold,
            'hard_rules': {
                'R001': 'days_past_due_current > 60 → DECLINE',
                'R002': 'credit_bureau_score < 300 → DECLINE',
                'R003': 'previous_default=Yes AND previous_loan_count > 1 → DECLINE',
                'R004': 'loan_amount / annual_revenue > 5 → REFER',
                'R005': 'owner_age < 18 → DECLINE',
                'R006': 'previous_default=Yes AND credit_bureau_score < 450 → DECLINE',
            }
        }
