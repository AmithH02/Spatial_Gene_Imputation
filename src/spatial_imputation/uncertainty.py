"""TISSUE uncertainty calibration and checks of whether it tracks real error.

TISSUE writes, for a prediction key ``k`` (e.g. ``"spage_predictions"``):
``obsm[k + '_uncertainty']`` (cell-centred variability), ``_error``, ``_score``,
``_groups`` (stratified calibration groups) and ``_lo`` / ``_hi`` (prediction interval).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as st


def calibrate(adata, pred_key: str, alpha_level: float = 0.23, n_neighbors: int = 15,
              k: int = 4, k2: int = 2) -> None:
    """Build the spatial graph (if missing), conformalise uncertainty and fit prediction intervals.

    Uses all genes as calibration genes (every gene has LOO predictions).
    ``alpha_level=0.23`` gives 77% prediction intervals, as in the reported analysis.
    """
    import tissue.main

    if "spatial_connectivities" not in adata.obsp:
        tissue.main.build_spatial_graph(adata, method="fixed_radius", n_neighbors=n_neighbors)
    tissue.main.conformalize_spatial_uncertainty(
        adata, pred_key, calib_genes=adata.var_names,
        grouping_method="kmeans_gene_cell", k=k, k2=k2,
    )
    tissue.main.conformalize_prediction_interval(
        adata, pred_key, symmetric=False, calib_genes=adata.var_names, alpha_level=alpha_level,
    )


def _frame(adata, key) -> pd.DataFrame:
    v = adata.obsm[key]
    return v if isinstance(v, pd.DataFrame) else pd.DataFrame(v, index=adata.obs_names, columns=adata.var_names)


def pi_width(adata, pred_key: str) -> pd.DataFrame:
    return _frame(adata, f"{pred_key}_hi") - _frame(adata, f"{pred_key}_lo")


def error_vs_uncertainty(adata, pred_key: str, uncertainty: str = "pi_width") -> pd.Series:
    """Per-gene Spearman rho between |error| and an uncertainty estimate.

    ``uncertainty`` is ``"pi_width"`` (prediction-interval width) or ``"ccv"``
    (TISSUE cell-centred variability). A reliable uncertainty gives rho > 0.
    """
    err = _frame(adata, f"{pred_key}_error").abs()
    unc = pi_width(adata, pred_key) if uncertainty == "pi_width" else _frame(adata, f"{pred_key}_uncertainty")
    rho = {g: st.spearmanr(err[g], unc[g])[0] for g in err.columns}
    return pd.Series(rho, name=f"rho_error_{uncertainty}")


def calibration_scores(adata, pred_key: str) -> pd.DataFrame:
    """Long frame of conformal scores and their calibration group (for KDE plots)."""
    scores = np.asarray(_frame(adata, f"{pred_key}_score")).ravel()
    groups = np.asarray(_frame(adata, f"{pred_key}_groups")).ravel()
    df = pd.DataFrame({"calibration_score": scores, "group": groups}).dropna()
    df["group"] = df["group"].astype(int).astype("category")
    return df


def reliability_quartiles(reliability: pd.Series, properties: pd.DataFrame,
                          props=("moran_i", "sparsity_sp")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split genes into quartiles of uncertainty reliability and test gene-property trends.

    Returns ``(df, tests)``: ``df`` has a ``quartile`` column; ``tests`` has the Spearman
    trend across quartiles and a Q1-vs-Q4 Mann-Whitney p-value for each property.
    """
    df = properties.join(reliability.rename("reliability"), how="inner").dropna(subset=["reliability"])
    df["quartile"] = pd.qcut(df["reliability"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    codes = df["quartile"].cat.codes
    rows = []
    for p in props:
        d = df[[p]].assign(q=codes).dropna()
        rho, p_trend = st.spearmanr(d["q"], d[p])
        q1, q4 = d.loc[d["q"] == 0, p], d.loc[d["q"] == 3, p]
        rows.append({"property": p, "trend_rho": rho, "trend_p": p_trend,
                     "q1_vs_q4_p": st.mannwhitneyu(q1, q4).pvalue})
    return df, pd.DataFrame(rows)
