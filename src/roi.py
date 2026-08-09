import numpy as np
import pandas as pd

from src.metrics import _rank_mask, _slice_lift


def _evaluate_strategy(name, mask, conversion, spend, treatment, email_cost, profit_per_conversion, note=None):
    """Evaluate one targeting policy given its boolean `mask` (who gets the email
    under this policy). Profit/spend lift is the REALIZED treated-vs-control lift
    on conversion/spend within the targeted subgroup (see module docstring below
    for why), not tau_hat treated as a dollar value.

    Both the benefit (incremental conversions/spend) and the cost (email cost) are
    projected over the FULL targeted population `n_targeted`, not the historically
    treated subset `n_treated`. Under randomization the per-customer lift rate is
    an unbiased effect estimate for the whole subgroup, and a deployed policy would
    email every customer matching the rule -- so charging cost on `n_targeted`
    while crediting benefit only on `n_treated` (the ~2/3 historically treated
    here) would understate the benefit relative to the cost.
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
            "incremental_conversion_rate_ci_lower": np.nan,
            "incremental_conversion_rate_ci_upper": np.nan,
            "incremental_conversions": 0.0,
            "incremental_profit": 0.0,
            "incremental_spend": 0.0,
            "net_value": 0.0,
            "net_value_per_customer": 0.0,
            "note": note,
        }

    conv_lift = _slice_lift(conversion, treatment, mask)
    spend_lift = _slice_lift(spend, treatment, mask)

    incremental_conversions = conv_lift["actual_lift"] * n_targeted
    incremental_profit = incremental_conversions * profit_per_conversion
    incremental_spend = spend_lift["actual_lift"] * n_targeted
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
        "incremental_conversion_rate_ci_lower": conv_lift["actual_lift_ci_lower"],
        "incremental_conversion_rate_ci_upper": conv_lift["actual_lift_ci_upper"],
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
    response_score,
    email_cost: float = 0.05,
    profit_per_conversion: float = 50.00,
    top_k: float = 0.3,
):
    """Constant-volume targeting comparison: hold the number of contacted customers
    FIXED (the top `top_k` fraction) and compare two SELECTION RULES against each
    other -- ranking customers by predicted response probability vs. ranking them
    by predicted uplift.

    Both policies contact the same number of customers at the same email cost, so
    any difference in realized lift or net value is due purely to *which* customers
    each rule selects. This deliberately isolates the value of uplift modeling from
    the separate, assumption-dependent question of *how many* customers to email
    (which depends on unobservable per-email costs) -- that "send everyone vs. a
    subset" sweep is intentionally not modeled here.

    Methodology: we cannot compute a per-customer counterfactual profit under a
    hypothetical policy assignment that differs from what a customer actually
    received. Instead, for each rule we define the targeted subgroup (top `top_k`
    by that rule's score) and compute the REALIZED treated-vs-control lift on
    conversion/spend within that subgroup (via src.metrics._slice_lift), with a
    95% CI so significance can be read directly. This is valid because treatment
    stays randomized within any subgroup defined by a pre-treatment score or model
    prediction -- the same logic underlying the decile table and Qini curve.

    OUTCOME NOTE (documented proxy -- read before interpreting): the two ranking
    signals and the measured outcome are NOT the same variable. BOTH rankers are
    defined on the project's primary outcome, `visit`: `tau_hat` is a visit-uplift
    estimate and `response_score` is a visit-probability estimate. The realized
    lift and net_value computed below, however, are measured on `conversion` (and
    `spend`) -- the business outcome. So this is an apples-to-apples RANKING
    comparison (both signals visit-based) evaluated on conversion; it is NOT a
    claim that either signal was trained on conversion. Ranking for conversion
    targeting via visit-based signals assumes visit-persuadability correlates with
    conversion-persuadability (plausible, since a visit is upstream of a
    conversion, but unverified here). A conversion-trained uplift model would be
    the proper tool and is a documented follow-up. tau_hat is used purely as a
    ranking signal, never as a dollar value. profit_per_conversion and email_cost
    are shared, identical dollar assumptions across both rules (so net_value
    differences reflect selection, not cost).

    Rules compared (both target the same top `top_k` fraction of customers):
      1. target_by_response_probability -- top `top_k` by `response_score`
      2. target_by_predicted_uplift     -- top `top_k` by `tau_hat`

    Returns a 2-row pandas.DataFrame, one row per rule, in the order above.
    """
    treatment = np.asarray(treatment)
    conversion = np.asarray(conversion)
    spend = np.asarray(spend)
    tau_hat = np.asarray(tau_hat)
    response_score = np.asarray(response_score)

    rows = [
        _evaluate_strategy(
            "target_by_response_probability", _rank_mask(response_score, top_k),
            conversion, spend, treatment, email_cost, profit_per_conversion,
            note=f"top {top_k:.0%} by response_score",
        ),
        _evaluate_strategy(
            "target_by_predicted_uplift", _rank_mask(tau_hat, top_k),
            conversion, spend, treatment, email_cost, profit_per_conversion,
            note=f"top {top_k:.0%} by tau_hat",
        ),
    ]

    return pd.DataFrame(rows).set_index("strategy")
