"""Statistical tests: permutation significance, method/dataset comparisons, Moran's I."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.stats.multitest import multipletests

from .metrics import LOWER_IS_BETTER


def _standardise(M: np.ndarray) -> np.ndarray:
    """Column-wise z-score (population SD). Constant columns become NaN."""
    M = np.asarray(M, float)
    sd = M.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (M - M.mean(axis=0)) / np.where(sd > 0, sd, np.nan)


def columnwise_pearson(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson r between matching columns of two (cells x genes) matrices."""
    return np.mean(_standardise(A) * _standardise(B), axis=0)


def permutation_test(predicted: pd.DataFrame, actual: pd.DataFrame, n_permutations: int = 999,
                     alpha: float = 0.05, seed: int | None = 0) -> pd.DataFrame:
    """One-sided permutation test of per-gene Pearson r between prediction and truth.

    The null shuffles cells (rows) of the prediction matrix; the same permutation is
    applied to all genes, as in the original analysis. Returns a frame with
    ``pearson_r``, ``p_value``, ``fdr_p_value`` (Benjamini-Hochberg) and ``significant_fdr``.
    """
    genes = predicted.columns.intersection(actual.columns)
    Zp = _standardise(predicted[genes].to_numpy())
    Za = _standardise(actual[genes].to_numpy())
    observed = np.mean(Zp * Za, axis=0)

    rng = np.random.default_rng(seed)
    exceed = np.zeros(len(genes))
    for _ in range(n_permutations):
        null = np.mean(Zp[rng.permutation(len(Zp))] * Za, axis=0)
        exceed += null >= observed
    p = (exceed + 1) / (n_permutations + 1)

    out = pd.DataFrame({"pearson_r": observed, "p_value": p}, index=genes)
    ok = np.isfinite(observed)
    out["fdr_p_value"] = np.nan
    out["significant_fdr"] = False
    if ok.any():
        reject, q, _, _ = multipletests(p[ok], alpha=alpha, method="fdr_bh")
        out.loc[ok, "fdr_p_value"] = q
        out.loc[ok, "significant_fdr"] = reject
    return out


def _direction(a_name, b_name, median_diff, metric):
    a_better = median_diff < 0 if metric in LOWER_IS_BETTER else median_diff > 0
    return f"{a_name} > {b_name}" if a_better else f"{b_name} > {a_name}"


def paired_method_test(a: pd.DataFrame, b: pd.DataFrame, metrics, a_name="Tangram", b_name="SpaGE",
                       alpha: float = 0.05) -> pd.DataFrame:
    """Wilcoxon signed-rank test per metric on genes scored by both methods."""
    rows = []
    for m in metrics:
        d = pd.concat([a[m].rename("a"), b[m].rename("b")], axis=1).dropna()
        w, p = st.wilcoxon(d["a"], d["b"])
        diff = (d["a"] - d["b"]).median()
        rows.append({"metric": m, "n_genes": len(d), "W": w, "p_value": p, "median_diff": diff,
                     "better": _direction(a_name, b_name, diff, m) if p < alpha else "n.s."})
    return pd.DataFrame(rows)


def unpaired_test(a: pd.DataFrame, b: pd.DataFrame, metrics, a_name="D1", b_name="D2",
                  alpha: float = 0.05) -> pd.DataFrame:
    """Two-sided Mann-Whitney U test per metric (e.g. dataset 1 vs dataset 2)."""
    rows = []
    for m in metrics:
        x, y = a[m].dropna(), b[m].dropna()
        u, p = st.mannwhitneyu(x, y, alternative="two-sided")
        diff = x.median() - y.median()
        rows.append({"metric": m, "U": u, "p_value": p, "median_diff": diff,
                     "better": _direction(a_name, b_name, diff, m) if p < alpha else "n.s."})
    return pd.DataFrame(rows)


def set_overlap(a, b) -> dict:
    a, b = set(a), set(b)
    union = len(a | b)
    return {"shared": len(a & b), "only_a": len(a - b), "only_b": len(b - a),
            "jaccard": len(a & b) / union if union else 0.0}


def morans_i(coords: np.ndarray, X: np.ndarray, k: int = 6) -> np.ndarray:
    """Moran's I for each column of ``X`` using a row-standardised k-NN weight matrix.

    Equivalent to ``esda.Moran(x, libpysal.weights.KNN(coords, k), transform='r').I``
    (self excluded from neighbours), vectorised over genes.
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, float)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)
    nbrs = idx[:, 1:]                      # drop self
    Z = X - X.mean(axis=0)
    lag = Z[nbrs].mean(axis=1)             # row-standardised spatial lag
    denom = (Z**2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Z * lag).sum(axis=0) / np.where(denom > 0, denom, np.nan)


def correlation_table(df: pd.DataFrame, pairs, group_col: str = "label", method: str = "pearson") -> pd.DataFrame:
    """Correlation (r, p) for each (x, y) pair within each group of ``df``."""
    fn = st.pearsonr if method == "pearson" else st.spearmanr
    rows = []
    for grp, sub in df.groupby(group_col, sort=False):
        for x, y in pairs:
            d = sub[[x, y]].dropna()
            if len(d) > 2:
                r, p = fn(d[x], d[y])
                rows.append({group_col: grp, "x": x, "y": y, "r": r, "p_value": p})
    return pd.DataFrame(rows)
