import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from src.config import FIGURES_DIR
from src.metrics import get_cumulative_gain_curve, get_qini_curve, propensity_matches_known


def save_fig(fig, filename, dpi=150, show=True, close=True):
    """Ensures FIGURES_DIR exists, saves `fig` as a PNG at FIGURES_DIR/filename,
    displays it inline (so it renders in the notebook regardless of cell output
    settings), and returns the saved path. All notebook figures should be saved
    through this helper so every plot lands in the same place with the same
    conventions.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        display(fig)
    if close:
        plt.close(fig)
    return path


def _plot_curve_with_random_baseline(curve_df, xlabel, ylabel, title, figsize):
    """Shared rendering for Qini/cumulative-gain curves. All columns returned by
    causalml.metrics.get_qini/get_cumgain converge to the same value at the full
    population (it's an order-invariant total), so the random-targeting baseline
    is just the diagonal from (first index, 0) to (last index, that shared value).
    """
    fig, ax = plt.subplots(figsize=figsize)
    for col in curve_df.columns:
        ax.plot(curve_df.index, curve_df[col], label=col)

    first_idx, last_idx = curve_df.index[0], curve_df.index[-1]
    endpoint = curve_df.iloc[-1, 0]
    ax.plot([first_idx, last_idx], [0, endpoint], "k--", label="Random")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig


def plot_qini_curve(y, treatment, tau_hat_dict, normalize=False, figsize=(8, 8)):
    """Qini curve for one or more models/variants, plus a random-targeting
    diagonal baseline. Computed via causalml.metrics.get_qini (through
    src.metrics.get_qini_curve) and rendered with matplotlib directly, since
    causalml.metrics.plot_qini renders its own figure and cannot be saved
    through `save_fig`. Returns the Figure for the caller to save.
    """
    qini_df = get_qini_curve(y, treatment, tau_hat_dict, normalize=normalize)
    return _plot_curve_with_random_baseline(
        qini_df, "Number of customers targeted", "Qini", "Qini Curve", figsize
    )


def plot_cumulative_gain_curve(y, treatment, tau_hat_dict, normalize=False, figsize=(8, 8)):
    """Cumulative gain curve, same pattern as plot_qini_curve but backed by
    causalml.metrics.get_cumgain (through src.metrics.get_cumulative_gain_curve).
    """
    gain_df = get_cumulative_gain_curve(y, treatment, tau_hat_dict, normalize=normalize)
    return _plot_curve_with_random_baseline(
        gain_df, "Number of customers targeted", "Cumulative gain", "Cumulative Gain Curve", figsize
    )


def plot_decile_lift_bar(
    decile_table, title="Actual lift by predicted-uplift decile",
    xlabel="Decile (1 = highest predicted uplift)", figsize=(8, 5),
):
    """Bar chart of actual_lift per row (e.g. decile or segment table), excluding
    the "Overall" row, with a horizontal reference line at the Overall row's
    actual_lift (the ATE). Error bars show the 95% CI on each row's lift (from
    actual_lift_se), so visibly overlapping bars signal that adjacent rows aren't
    reliably distinguishable from noise alone -- not necessarily a ranking failure.
    Works for both build_decile_table and build_segment_table output (same shape).
    """
    rows = decile_table.drop(index="Overall")
    ate = decile_table.loc["Overall", "actual_lift"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        rows.index.astype(str), rows["actual_lift"],
        yerr=1.96 * rows["actual_lift_se"], capsize=4, color="steelblue",
    )
    ax.axhline(ate, color="black", linestyle="--", label=f"Overall ATE ({ate:.4f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Actual lift (treated - control outcome rate)")
    ax.set_title(f"{title}\n(error bars: 95% CI)")
    ax.legend()
    return fig


def plot_propensity_diagnostic(p_hat, known_p, tolerance=0.01, figsize=(7.5, 5)):
    """Histogram of estimated propensity scores against the known randomized-design
    propensity. The x-axis is zoomed to the 1st-99th percentile band of `p_hat` so
    the bulk of the distribution is visible even when a handful of outliers span a
    much wider range (points outside that band are counted, not dropped -- noted in
    a corner annotation instead). Mean and known propensity are marked, and the
    title reports an explicit PASS/FAIL verdict from `propensity_matches_known`, so
    agreement is visible at a glance rather than something the reader has to eyeball.
    """
    p_hat = np.asarray(p_hat)
    check = propensity_matches_known(p_hat, known_p, tolerance=tolerance)

    lo, hi = np.percentile(p_hat, [1, 99])
    n_outliers = int(((p_hat < lo) | (p_hat > hi)).sum())

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(p_hat, bins=50, color="steelblue", alpha=0.8)
    ax.axvline(known_p, color="black", linestyle="--", label=f"Known p = {known_p:.4f}")
    ax.axvline(check["mean_p"], color="darkorange", linestyle=":", label=f"Estimated mean = {check['mean_p']:.4f}")
    ax.set_xlim(lo, hi)

    verdict = "PASS" if check["agrees"] else "FAIL"
    color = "seagreen" if check["agrees"] else "firebrick"
    ax.set_title(
        f"Estimated vs. Known Propensity -- {verdict}\n"
        f"|diff| = {check['diff']:.4f}  (tolerance {tolerance:.3f})  |  std = {check['std_p']:.4f}",
        color=color,
    )
    ax.set_xlabel("Estimated propensity (zoomed to 1st-99th percentile)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left")
    ax.text(
        0.98, 0.98, f"{n_outliers} points\noutside view",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color="dimgray",
    )
    fig.tight_layout()
    return fig
