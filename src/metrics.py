import numpy as np
import pandas as pd
from causalml.metrics import auuc_score, get_cumgain, get_qini, qini_score
from sklearn.model_selection import StratifiedKFold


def stratified_uplift_folds(y, treatment, n_splits=5, random_state=42):
    """Return train/validation indices for uplift out-of-fold evaluation.

    The folds are stratified on the joint treatment/outcome label rather than on
    treatment alone.  This keeps treated/control responders and non-responders
    represented in every fold, which is important for sparse binary outcomes.
    Only the training split should be passed to this helper; the held-out test
    set must never participate in model or hyperparameter selection.
    """
    y = np.asarray(y)
    treatment = np.asarray(treatment)

    if len(y) != len(treatment):
        raise ValueError("y and treatment must have the same length.")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("y must be binary for joint-stratified uplift folds.")
    if not set(np.unique(treatment)).issubset({0, 1}):
        raise ValueError("treatment must be binary for uplift evaluation.")

    joint_strata = treatment.astype(int) * 2 + y.astype(int)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    return splitter.split(np.zeros(len(y)), joint_strata)


def validate_prediction_frame(
    frame,
    y_true,
    treatment=None,
    *,
    frame_name="prediction frame",
    row_id_col=None,
):
    """Fail fast when a saved prediction file is not aligned to a data split.

    Row counts alone cannot detect stale or reordered prediction files.  The
    saved ``y_true`` (and, when supplied, ``treatment``) vectors provide a small
    deterministic fingerprint of the canonical train/test split.  A sequential
    row identifier can also be checked when a model writes one.
    """
    if "y_true" not in frame.columns:
        raise ValueError(f"{frame_name} must include a y_true column.")

    expected_y = np.asarray(y_true)
    if len(frame) != len(expected_y):
        raise ValueError(
            f"{frame_name} has {len(frame)} rows; expected {len(expected_y)}."
        )
    if not np.array_equal(frame["y_true"].to_numpy(), expected_y):
        raise ValueError(f"{frame_name} y_true does not match the canonical split.")

    if treatment is not None:
        if "treatment" not in frame.columns:
            raise ValueError(f"{frame_name} must include a treatment column.")
        expected_treatment = np.asarray(treatment)
        if not np.array_equal(frame["treatment"].to_numpy(), expected_treatment):
            raise ValueError(
                f"{frame_name} treatment does not match the canonical split."
            )

    if row_id_col is not None:
        if row_id_col not in frame.columns:
            raise ValueError(f"{frame_name} must include {row_id_col}.")
        expected_ids = np.arange(len(frame))
        if not np.array_equal(frame[row_id_col].to_numpy(), expected_ids):
            raise ValueError(
                f"{frame_name} {row_id_col} is not the canonical sequential row id."
            )

    return True


def _rank_mask(tau_hat, top_fraction):
    """Boolean mask selecting the top `top_fraction` of rows by descending tau_hat.

    Ties are broken by original row order (stable), so the returned mask always
    selects exactly ceil(top_fraction * n) rows.
    """
    n = len(tau_hat)
    ranks = pd.Series(np.asarray(tau_hat)).rank(method="first", ascending=False)
    cutoff = int(np.ceil(top_fraction * n))
    return (ranks <= cutoff).to_numpy()


def _slice_lift(y, treatment, mask):
    """Realized treated-vs-control outcome rate and lift within the subset where mask is True,
    plus a 95% CI on the lift from the standard errors of the two group means.

    Valid under randomization: treatment stays random within any subgroup defined
    by a pre-treatment score, even if the subgroup itself was chosen using that score.
    Treated/control samples are independent within the subgroup, so the lift's
    variance is just the sum of the two groups' variances. Uses the empirical
    variance of `y` within each group (not the binomial p*(1-p) shortcut), since
    `_slice_lift` is also reused for continuous outcomes like `spend` in
    src.roi, where p*(1-p) isn't a valid variance formula.
    """
    y = np.asarray(y)[mask]
    t = np.asarray(treatment)[mask]
    n_treated, n_control = int(t.sum()), int((1 - t).sum())
    y_treated, y_control = y[t == 1], y[t == 0]
    treated_rate = y_treated.mean() if n_treated > 0 else np.nan
    control_rate = y_control.mean() if n_control > 0 else np.nan
    actual_lift = treated_rate - control_rate

    se_treated = np.sqrt(y_treated.var(ddof=1) / n_treated) if n_treated > 1 else np.nan
    se_control = np.sqrt(y_control.var(ddof=1) / n_control) if n_control > 1 else np.nan
    lift_se = np.sqrt(se_treated**2 + se_control**2)

    return {
        "n_customers": int(mask.sum()),
        "n_treated": n_treated,
        "n_control": n_control,
        "treated_outcome_rate": treated_rate,
        "control_outcome_rate": control_rate,
        "actual_lift": actual_lift,
        "actual_lift_se": lift_se,
        "actual_lift_ci_lower": actual_lift - 1.96 * lift_se,
        "actual_lift_ci_upper": actual_lift + 1.96 * lift_se,
    }


def build_decile_table(y, treatment, tau_hat, n_bins=10):
    """Uplift-by-decile table: rank customers by descending tau_hat, split into
    `n_bins` equal-count bins (bin 1 = highest predicted uplift, bin n_bins =
    lowest), and report treated/control counts, treated/control outcome rates,
    and actual lift per bin. Appends an "Overall" row (the ATE) for reference.
    """
    n = len(tau_hat)
    ranks = pd.Series(np.asarray(tau_hat)).rank(method="first", ascending=False)
    bins = np.ceil(ranks / n * n_bins).astype(int).clip(1, n_bins)

    rows = []
    for b in range(1, n_bins + 1):
        mask = (bins == b).to_numpy()
        row = _slice_lift(y, treatment, mask)
        row["decile"] = b
        row["mean_tau_hat"] = float(np.asarray(tau_hat)[mask].mean())
        rows.append(row)

    overall_mask = np.ones(n, dtype=bool)
    overall = _slice_lift(y, treatment, overall_mask)
    overall["decile"] = "Overall"
    overall["mean_tau_hat"] = float(np.mean(tau_hat))
    rows.append(overall)

    table = pd.DataFrame(rows).set_index("decile")
    cols = [
        "n_customers",
        "n_treated",
        "n_control",
        "treated_outcome_rate",
        "control_outcome_rate",
        "actual_lift",
        "actual_lift_se",
        "actual_lift_ci_lower",
        "actual_lift_ci_upper",
        "mean_tau_hat",
    ]
    return table[cols]


def build_segment_table(y, treatment, segment, segment_order=None, tau_hat=None):
    """Realized treated-vs-control lift within each level of a categorical `segment`
    label (e.g. a customer attribute), with the same per-level CI as build_decile_table.
    Appends an "Overall" row for reference.

    Complements decile-based evaluation: deciles assume smoothly-varying continuous
    heterogeneity, ranked and cut into equal-count bins. Real heterogeneity can instead
    be concentrated in a specific categorical subgroup that doesn't align with decile
    boundaries -- a strong small segment gets split across bins and diluted with
    surrounding customers, understating how much real signal actually exists. If
    `tau_hat` is provided, includes each segment's mean predicted uplift alongside its
    actual lift, making under/over-prediction directly visible per segment.
    """
    segment = pd.Series(np.asarray(segment))
    n = len(segment)
    levels = segment_order if segment_order is not None else sorted(segment.unique())

    rows = []
    for level in levels:
        mask = (segment == level).to_numpy()
        row = _slice_lift(y, treatment, mask)
        row["segment"] = level
        if tau_hat is not None:
            row["mean_tau_hat"] = float(np.asarray(tau_hat)[mask].mean())
        rows.append(row)

    overall_mask = np.ones(n, dtype=bool)
    overall = _slice_lift(y, treatment, overall_mask)
    overall["segment"] = "Overall"
    if tau_hat is not None:
        overall["mean_tau_hat"] = float(np.mean(tau_hat))
    rows.append(overall)

    table = pd.DataFrame(rows).set_index("segment")
    cols = [
        "n_customers",
        "n_treated",
        "n_control",
        "treated_outcome_rate",
        "control_outcome_rate",
        "actual_lift",
        "actual_lift_se",
        "actual_lift_ci_lower",
        "actual_lift_ci_upper",
    ]
    if tau_hat is not None:
        cols.append("mean_tau_hat")
    return table[cols]


def segments_agree_across_samples(table_a, table_b, segment, z_threshold=1.96):
    """Compare one segment's actual_lift estimate across two independently-computed
    build_segment_table outputs (e.g. from train vs. test), via a two-sample z-test on
    the difference. Used to check whether a segment effect "discovered" by searching
    across many candidate features/segments in a single sample actually replicates in
    an independent sample, or was inflated by that search -- picking whichever segment
    looks most extreme in one sample biases that sample's estimate upward on average
    (a selection/winner's-curse effect), even though the segment itself may be real.
    """
    a, b = table_a.loc[segment], table_b.loc[segment]
    diff = float(b["actual_lift"] - a["actual_lift"])
    se_diff = float(np.sqrt(a["actual_lift_se"] ** 2 + b["actual_lift_se"] ** 2))
    z = diff / se_diff
    return {
        "segment": segment,
        "estimate_a": float(a["actual_lift"]),
        "estimate_b": float(b["actual_lift"]),
        "diff": diff,
        "se_diff": se_diff,
        "z": float(z),
        "agrees": bool(abs(z) < z_threshold),
    }


def build_uplift_eval_frame(y, treatment, tau_hat_dict):
    """Builds the DataFrame shape causalml.metrics functions require: column 'y'
    for the outcome, column 'w' for treatment (0/1), and one column per entry in
    tau_hat_dict (key = model/variant name, value = tau_hat array).
    """
    df = pd.DataFrame({"y": np.asarray(y), "w": np.asarray(treatment)})
    for name, tau in tau_hat_dict.items():
        df[name] = np.asarray(tau)
    return df


def compute_qini_auuc(y, treatment, tau_hat_dict, normalize=True):
    """Thin wrapper around causalml.metrics.qini_score / auuc_score. Returns
    {"qini_score": pd.Series, "auuc_score": pd.Series} keyed by model name.
    """
    df = build_uplift_eval_frame(y, treatment, tau_hat_dict)
    return {
        "qini_score": qini_score(df, outcome_col="y", treatment_col="w", normalize=normalize),
        "auuc_score": auuc_score(df, outcome_col="y", treatment_col="w", normalize=normalize),
    }


def get_cumulative_gain_curve(y, treatment, tau_hat_dict, normalize=False):
    """Thin wrapper around causalml.metrics.get_cumgain; returns the raw cumulative-gain
    DataFrame (index = cumulative population count, columns = model names) for plotting.
    """
    df = build_uplift_eval_frame(y, treatment, tau_hat_dict)
    return get_cumgain(df, outcome_col="y", treatment_col="w", normalize=normalize)


def get_qini_curve(y, treatment, tau_hat_dict, normalize=False):
    """Thin wrapper around causalml.metrics.get_qini; same shape as get_cumulative_gain_curve."""
    df = build_uplift_eval_frame(y, treatment, tau_hat_dict)
    return get_qini(df, outcome_col="y", treatment_col="w", normalize=normalize)


def propensity_matches_known(p_hat, known_p, tolerance=0.01):
    """Compare an estimated propensity's mean against the known randomized-design
    propensity. Returns the comparison stats plus a boolean verdict, used to decide
    whether a data-driven propensity model has recovered the known assignment rate
    closely enough to skip fitting a redundant model variant on top of it.
    """
    p_hat = np.asarray(p_hat)
    mean_p = float(p_hat.mean())
    diff = abs(mean_p - known_p)
    return {
        "known_p": known_p,
        "mean_p": mean_p,
        "std_p": float(p_hat.std()),
        "diff": diff,
        "tolerance": tolerance,
        "agrees": diff < tolerance,
    }


def uplift_at_top_k(y, treatment, tau_hat, k):
    """Realized treated-vs-control lift within the top-k fraction of customers
    ranked by descending tau_hat. Also returns incremental_outcomes_captured,
    the extra outcomes attributable to treated customers in this slice relative
    to the control rate. Call for k in (0.1, 0.2, 0.3) to get uplift at top 10/20/30%.
    """
    mask = _rank_mask(tau_hat, k)
    result = _slice_lift(y, treatment, mask)
    result["k"] = k
    result["incremental_outcomes_captured"] = result["actual_lift"] * result["n_treated"]
    return result
