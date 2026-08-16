import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import display
from src.config import FIGURES_DIR
from src.metrics import get_cumulative_gain_curve, get_qini_curve, compute_qini_auuc, propensity_matches_known


def save_fig(fig, filename, dpi=150, show=True, close=True):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        display(fig)
    if close:
        plt.close(fig)
    return path


def _plot_curve_with_random_baseline(curve_df, xlabel, ylabel, title, figsize, score_labels=None):
    n = curve_df.index.max()
    x = curve_df.index / n  # normalize x-axis to 0-1

    colors = cm.tab10.colors + cm.tab20.colors  # enough distinct colors for 10+ models
    fig, ax = plt.subplots(figsize=figsize)
    for i, col in enumerate(curve_df.columns):
        label = col
        if score_labels is not None and col in score_labels:
            label = f"{col} ({score_labels[col]:.4f})"
        ax.plot(x, curve_df[col], label=label, linewidth=1.5, color=colors[i % len(colors)])

    endpoint = curve_df.iloc[-1, 0]
    ax.plot([0, 1], [0, endpoint], "k--", label="Random", linewidth=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.tight_layout()
    return fig


def plot_qini_curve(y, treatment, tau_hat_dict, normalize=True, figsize=(10, 8)):
    qini_df = get_qini_curve(y, treatment, tau_hat_dict, normalize=normalize)
    qini_scores = compute_qini_auuc(y, treatment, tau_hat_dict, normalize=normalize)["qini_score"]
    return _plot_curve_with_random_baseline(
        qini_df, "Fraction of population targeted", "Qini", "Qini Curve",
        figsize, score_labels=qini_scores.to_dict(),
    )


def plot_cumulative_gain_curve(y, treatment, tau_hat_dict, normalize=True, figsize=(10, 8)):
    """Cumulative gain (uplift/AUUC) curve, same pattern as plot_qini_curve but
    backed by causalml.metrics.get_cumgain (through src.metrics.get_cumulative_gain_curve).
    """
    gain_df = get_cumulative_gain_curve(y, treatment, tau_hat_dict, normalize=normalize)
    auuc_scores = compute_qini_auuc(y, treatment, tau_hat_dict, normalize=normalize)["auuc_score"]
    return _plot_curve_with_random_baseline(
        gain_df, "Fraction of population targeted", "Cumulative Gain", "Uplift Curve (AUUC)",
        figsize, score_labels=auuc_scores.to_dict(),
    )


def plot_decile_lift_bar(
    decile_table, title="Actual lift by predicted-uplift decile",
    xlabel="Decile (1 = highest predicted uplift)", figsize=(8, 5),
):
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