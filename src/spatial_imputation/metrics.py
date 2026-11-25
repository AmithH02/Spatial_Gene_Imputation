"""Per-gene agreement metrics between predicted and measured expression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as st

# names used in tables and figures
METRIC_LABELS = {
    "pearson_r": "PCC",
    "spearman_r": "Spearman",
    "RMSE": "RMSE",
    "cosine_sim": "Cosine",
    "js": "JS",
    "ssim": "SSIM",
}
LOWER_IS_BETTER = {"RMSE", "js"}


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def cosine_similarity(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return float(x @ y / (nx * ny)) if nx > 0 and ny > 0 else 0.0


def js_divergence(p, q) -> float:
    """Jensen-Shannon divergence (natural log) between two non-negative vectors.

    Each vector is normalised to sum to 1. Entries <= 0 contribute nothing, as in
    the original analysis. Returns NaN if either vector sums to 0.
    """
    p = np.array(p, dtype=float)  # copies, so inputs are never modified
    q = np.array(q, dtype=float)
    ps, qs = p.sum(), q.sum()
    if ps == 0 or qs == 0:
        return np.nan
    p /= ps
    q /= qs
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def ssim_vector(x, y, c1: float = 0.01, c2: float = 0.03) -> float:
    """Global SSIM between two expression vectors, each scaled to max 1.

    Kept identical to the reported analysis: population variance for the
    sigma terms and sample covariance (``np.cov``) for the cross term.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x / x.max() if x.max() > 0 else x
    y = y / y.max() if y.max() > 0 else y
    mx, my = x.mean(), y.mean()
    num = (2 * mx * my + c1) * (2 * np.cov(x, y)[0, 1] + c2)
    den = (mx**2 + my**2 + c1) * (x.var() + y.var() + c2)
    return float(num / den) if den > 0 else np.nan


def gene_metrics(predicted: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Pearson, Spearman, RMSE, cosine, JS and SSIM for every gene in both frames.

    ``predicted`` and ``actual`` are cells x genes with the same row order.
    """
    genes = predicted.columns.intersection(actual.columns)
    rows = {}
    for g in genes:
        x = predicted[g].to_numpy(dtype=float)
        y = actual[g].to_numpy(dtype=float)
        rows[g] = {
            "pearson_r": st.pearsonr(x, y)[0],
            "spearman_r": st.spearmanr(x, y)[0],
            "RMSE": rmse(y, x),
            "cosine_sim": cosine_similarity(x, y),
            "js": js_divergence(x, y),
            "ssim": ssim_vector(x, y),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def sparsity_weighted_score(scores: pd.Series, sparsity_sp: pd.Series) -> float:
    """Mean score weighted by (1 - spatial sparsity), as in the Tangram benchmark."""
    w = 1 - sparsity_sp
    return float((scores * w).sum() / w.sum())


def score_sparsity_auc(scores, sparsity_sp, deg: int = 2):
    """Tangram-style AUC: area under a quadratic fit of score vs spatial sparsity on [0, 1].

    Returns ``(auc, (fit_x, fit_y))``.
    """
    from sklearn.metrics import auc

    xs, ys = np.asarray(sparsity_sp, float), np.asarray(scores, float)
    pol = np.poly1d(np.polyfit(xs, ys, deg))
    px = np.linspace(0, 1, 10)
    py = np.minimum(pol(px), 1)
    roots = [np.real(r) for r in pol.r if np.isreal(r) and 0 <= np.real(r) <= 1]
    if roots:
        px, py = np.append(px, roots[0]), np.append(py, 0)
    px, py = np.append(px, 1), np.append(py, pol(1))
    pts = sorted((a, b) for a, b in zip(px, py) if 0 <= a <= 1 and 0 <= b <= 1)
    if len(pts) < 2:
        return np.nan, ([], [])
    fx, fy = zip(*pts)
    return float(auc(fx, fy)), (fx, fy)


def tidy_metrics(metrics: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Stack ``{(method, dataset): metrics_df}`` into one long frame with a ``label`` column."""
    from .config import METHOD_LABELS

    parts = []
    for (method, ds), df in metrics.items():
        d = df.copy()
        d["gene"] = d.index
        d["method"] = METHOD_LABELS.get(method, method)
        d["dataset"] = ds.upper()
        d["label"] = f"{d['method'].iat[0]} ({ds.upper()})"
        parts.append(d)
    return pd.concat(parts, ignore_index=True)
