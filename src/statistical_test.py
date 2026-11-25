# ─── Statistical Tests ────────────────────────────────────────────────────
import numpy as np
from sklearn.metrics import r2_score
from scipy import stats

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 metric=r2_score, n_boot=5000, ci=0.95, seed=42) -> dict:
    """Non-parametric bootstrap CI on a single metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[b] = metric(y_true[idx], y_pred[idx])

    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)

    return {"point": metric(y_true, y_pred), "ci_95": (lo, hi)}


def bootstrap_difference(y_true, y_pred_a, y_pred_b,
                         metric=r2_score, n_boot=5000, ci=0.95, seed=42) -> dict:
    """Bootstrap CI on the difference in metric between two models."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b] = metric(y_true[idx], y_pred_a[idx]) - metric(y_true[idx], y_pred_b[idx])

    lo = np.percentile(diffs, (1 - ci) / 2 * 100)
    hi = np.percentile(diffs, (1 + ci) / 2 * 100)
    point = metric(y_true, y_pred_a) - metric(y_true, y_pred_b)

    return {
        "point_diff": point,
        "ci_95": (lo, hi),
        "significant": not (lo <= 0 <= hi),
    }


def paired_t_test(scores_a: list, scores_b: list, alpha=0.05) -> dict:
    """Paired t-test on per-fold CV scores."""
    diffs = np.array(scores_a) - np.array(scores_b)
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    n = len(diffs)
    se = diffs.std(ddof=1) / np.sqrt(n)
    ci_lo = diffs.mean() - stats.t.ppf(1 - alpha / 2, n - 1) * se
    ci_hi = diffs.mean() + stats.t.ppf(1 - alpha / 2, n - 1) * se

    return {
        "mean_diff": diffs.mean(),
        "ci_95": (ci_lo, ci_hi),
        "t_stat": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
    }

def friedman_test(fold_scores: dict[str, list]) -> tuple:
    """Friedman test for >2 models. Returns (chi2, p_value)."""
    return stats.friedmanchisquare(*fold_scores.values())
