import numpy as np
import pandas as pd

from src.metrics import _rank_mask, _slice_lift


def _evaluate_strategy(name, mask, conversion, spend, treatment, email_cost, profit_per_conversion, note=None):
    """Evaluate one targeting policy given its boolean `mask` (who gets the email
    under this policy). Profit/spend lift is the REALIZED treated-vs-control lift
    on conversion/spend within the targeted subgroup (see module docstring below
    for why), not tau_hat treated as a dollar value.
    """
    n = len(mask)
    n_targeted = int(mask.sum())

    if n_targeted == 0:
        return {
            "strategy": name,
            "n_targeted": 0,
            "pct_targeted": 0.0,
            "email_cost_total": 0.0,
            "treated_outcome_rate": np.nan,
            "control_outcome_rate": np.nan,
            "incremental_conversion_rate": np.nan,
            "incremental_conversions": 0.0,
            "incremental_profit": 0.0,
            "incremental_spend": 0.0,
            "net_value": 0.0,
            "net_value_per_customer": 0.0,
            "note": note,
        }

    conv_lift = _slice_lift(conversion, treatment, mask)
    spend_lift = _slice_lift(spend, treatment, mask)

    incremental_conversions = conv_lift["actual_lift"] * conv_lift["n_treated"]
    incremental_profit = incremental_conversions * profit_per_conversion
    incremental_spend = spend_lift["actual_lift"] * spend_lift["n_treated"]
    email_cost_total = n_targeted * email_cost
    net_value = incremental_profit - email_cost_total

    return {
        "strategy": name,
        "n_targeted": n_targeted,
        "pct_targeted": n_targeted / n,
        "email_cost_total": email_cost_total,
        "treated_outcome_rate": conv_lift["treated_outcome_rate"],
        "control_outcome_rate": conv_lift["control_outcome_rate"],
        "incremental_conversion_rate": conv_lift["actual_lift"],
        "incremental_conversions": incremental_conversions,
        "incremental_profit": incremental_profit,
        "incremental_spend": incremental_spend,
        "net_value": net_value,
        "net_value_per_customer": net_value / n_targeted,
        "note": note,
    }


def compare_targeting_strategies(
    conversion,
    spend,
    tau_hat,
    treatment,
    response_score=None,
    email_cost: float = 0.05,
    profit_per_conversion: float = 50.00,
    top_k_response: float = 0.3,
    top_k_uplift: float = 0.3,
):
    """Compares email-targeting strategies using REALIZED held-out conversion/
    spend outcomes, not tau_hat treated directly as a dollar value.

    Methodology: we cannot compute a per-customer counterfactual profit under a
    hypothetical policy assignment that differs from what a customer actually
    received. Instead, for each policy we define the targeted subgroup by the
    policy rule, then compute the REALIZED treated-vs-control lift on
    conversion/spend within that subgroup (via src.metrics._slice_lift). This is
    valid because treatment stays randomized within any subgroup defined by a
    pre-treatment score or model prediction -- the same logic underlying the
    decile table and Qini curve.

    tau_hat is expected to be a visit-uplift estimate (the project's primary
    outcome). For the "target_by_expected_profit_positive" strategy, tau_hat is
    used as a documented proxy ranking signal for the profit decision rule, since
    no conversion-specific uplift model exists in this project yet; this is a
    ranking/thresholding convenience, not a claim that tau_hat is itself a dollar
    value. profit_per_conversion and email_cost are the only dollar assumptions
    applied (no separate incentive-cost line item, since the Hillstrom dataset
    has no such column).

    Strategies compared (mirrors README's Policy Value and ROI Simulation
    section):
      1. send_everyone
      2. send_no_one
      3. target_by_response_probability -- top `top_k_response` by
         `response_score` (skipped, with a note, if response_score is None)
      4. target_by_predicted_uplift -- top `top_k_uplift` by tau_hat
      5. target_by_expected_profit_positive -- tau_hat * profit_per_conversion
         - email_cost > 0

    Returns a pandas.DataFrame, one row per strategy, in the order above.
    """
    treatment = np.asarray(treatment)
    conversion = np.asarray(conversion)
    spend = np.asarray(spend)
    tau_hat = np.asarray(tau_hat)
    n = len(tau_hat)

    rows = []

    rows.append(
        _evaluate_strategy(
            "send_everyone", np.ones(n, dtype=bool), conversion, spend, treatment,
            email_cost, profit_per_conversion,
        )
    )
    rows.append(
        _evaluate_strategy(
            "send_no_one", np.zeros(n, dtype=bool), conversion, spend, treatment,
            email_cost, profit_per_conversion,
        )
    )

    if response_score is None:
        rows.append(
            _evaluate_strategy(
                "target_by_response_probability", np.zeros(n, dtype=bool), conversion,
                spend, treatment, email_cost, profit_per_conversion,
                note="skipped: response_score not provided",
            )
        )
    else:
        mask = _rank_mask(np.asarray(response_score), top_k_response)
        rows.append(
            _evaluate_strategy(
                "target_by_response_probability", mask, conversion, spend, treatment,
                email_cost, profit_per_conversion,
                note=f"top {top_k_response:.0%} by response_score",
            )
        )

    mask = _rank_mask(tau_hat, top_k_uplift)
    rows.append(
        _evaluate_strategy(
            "target_by_predicted_uplift", mask, conversion, spend, treatment,
            email_cost, profit_per_conversion,
            note=f"top {top_k_uplift:.0%} by tau_hat",
        )
    )

    expected_profit = tau_hat * profit_per_conversion - email_cost
    mask = expected_profit > 0
    rows.append(
        _evaluate_strategy(
            "target_by_expected_profit_positive", mask, conversion, spend, treatment,
            email_cost, profit_per_conversion,
            note="tau_hat (visit-uplift) used as a documented proxy for conversion-uplift",
        )
    )

    return pd.DataFrame(rows).set_index("strategy")
