"""Figures used in the notebooks and the write-up."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from .config import X_COL, Y_COL
from .metrics import METRIC_LABELS, score_sparsity_auc

METHOD_COLOURS = {"Tangram": "#5899C8", "SpaGE": "#EA6565"}
CONDITION_COLOURS = {
    "Tangram (D1)": "#b3cde3",
    "SpaGE (D1)": "#ccebc5",
    "Tangram (D2)": "#6497b1",
    "SpaGE (D2)": "#a1d99b",
}
DATASET_COLOURS = {"D1": "#E57373", "D2": "#64B5F6"}
CONDITION_ORDER = list(CONDITION_COLOURS)


def set_style(font_scale: float = 1.4) -> None:
    sns.set_theme(style="ticks", font_scale=font_scale)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def display_values(v, pct: float = 95) -> np.ndarray:
    """Clip negatives, log1p, and cap at the ``pct`` percentile (for spatial colour maps)."""
    v = np.log1p(np.clip(np.asarray(v, float).ravel(), 0, None))
    return np.minimum(v, np.percentile(v, pct))


def spatial_scatter(ax, adata, values, title: str = "", s: float = 1, cmap="viridis",
                    colorbar: bool = True, cbar_label: str = "log expression", **kw):
    ax.set_aspect("equal")
    ax.axis("off")
    im = ax.scatter(adata.obs[X_COL], adata.obs[Y_COL], c=values, s=s, cmap=cmap, rasterized=True, **kw)
    ax.set_title(title)
    if colorbar:
        cb = plt.colorbar(im, ax=ax, shrink=0.7)
        cb.set_label(cbar_label, rotation=270, labelpad=15)
    return im


def plot_cells(adata, title: str = "Cells kept after cortex filtering"):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(adata.obs[X_COL], adata.obs[Y_COL], s=1, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_title(f"{title} (n = {adata.n_obs:,})")
    ax.set_xlabel(X_COL)
    ax.set_ylabel(Y_COL)
    return fig


def plot_count_depth(spatial, reference):
    """Violin plot of transcripts per cell (log scale), spatial vs single-cell."""
    sp_counts = (spatial.obs["transcript_count"].to_numpy() if "transcript_count" in spatial.obs
                 else np.asarray(spatial.X.sum(axis=1)).ravel())
    sc_counts = np.asarray(reference.X.sum(axis=1)).ravel()
    df = pd.DataFrame({
        "counts": np.concatenate([sp_counts, sc_counts]),
        "dataset": ["Spatial"] * len(sp_counts) + ["Single-cell"] * len(sc_counts),
    })
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.violinplot(data=df, x="dataset", y="counts", cut=0, inner="box", ax=ax)
    ax.set_yscale("log")
    ax.set_ylabel("Counts per cell (log)")
    ax.set_xlabel("")
    return fig


def plot_actual_vs_predicted(adata, gene: str, pred_keys=("tangram_predictions", "spage_predictions"),
                             scores: dict | None = None, figsize_per=5):
    """Measured expression next to each method's LOO prediction for one gene."""
    panels = [("Measured", adata[:, gene].X)] + [
        (k.replace("_predictions", "").capitalize().replace("Spage", "SpaGE"), adata.obsm[k][gene].values)
        for k in pred_keys
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(figsize_per * len(panels), figsize_per))
    for ax, (name, vals) in zip(axes, panels):
        title = name if not scores or name not in scores else f"{name} (r = {scores[name]:.2f})"
        spatial_scatter(ax, adata, display_values(vals), title=title, colorbar=False)
    fig.suptitle(gene)
    fig.tight_layout()
    return fig


def plot_score_vs_sparsity(metrics: pd.DataFrame, score: str = "pearson_r", colour="coral"):
    """Score distribution and score vs single-cell / spatial / difference sparsity."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), sharey=True)
    sns.histplot(data=metrics, y=score, bins=10, ax=axes[0], color=colour)
    axes[0].set_title("Score distribution")
    for ax, col, title in zip(axes[1:], ["sparsity_sc", "sparsity_sp", "sparsity_diff"],
                              ["vs sparsity (single-cell)", "vs sparsity (spatial)", "vs |sp - sc|"]):
        sns.scatterplot(data=metrics, x=col, y=score, ax=ax, alpha=0.6, color=colour)
        ax.set_xlim(0, 1)
        ax.set_title(title)
    axes[0].set_ylim(0, 1)
    fig.tight_layout()
    return fig


def plot_sparsity_auc(metrics: pd.DataFrame, score: str = "pearson_r", title: str = ""):
    value, (fx, fy) = score_sparsity_auc(metrics[score], metrics["sparsity_sp"])
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(x=metrics["sparsity_sp"], y=metrics[score], alpha=0.5, ax=ax)
    ax.plot(fx, fy, c="r")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Spatial sparsity", ylabel=METRIC_LABELS.get(score, score),
           title=title)
    ax.text(0.03, 0.05, f"AUC = {value:.3f}", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
    return fig


def plot_metric_bars(tidy: pd.DataFrame, metrics=("pearson_r", "spearman_r", "js", "ssim")):
    """Mean +/- SD of each metric per condition (Figure 2b of the write-up)."""
    summary = tidy.groupby("label")[list(metrics)].agg(["mean", "std"])
    order = [c for c in CONDITION_ORDER if c in summary.index][::-1]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, m in zip(axes.flat, metrics):
        s = summary[m].loc[order]
        ax.barh(order, s["mean"], xerr=s["std"], color=[CONDITION_COLOURS[c] for c in order],
                capsize=5, error_kw={"alpha": 0.5})
        ax.set_xlim(0, 1)
        ax.set_xlabel(METRIC_LABELS.get(m, m))
        ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.suptitle("Imputation performance across metrics and datasets", weight="bold")
    return fig


def plot_significant_fraction(sig: pd.DataFrame):
    """Bar chart of the fraction of genes significant after FDR, per condition.

    ``sig`` is a tidy frame with ``label`` and boolean ``significant_fdr`` columns.
    """
    frac = sig.groupby("label")["significant_fdr"].mean().reindex(CONDITION_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(frac.index, frac.values, color=sns.color_palette("BuPu", len(frac)),
                  edgecolor="black", linewidth=1.2)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in frac.values], padding=3)
    ax.set_ylim(0, frac.max() * 1.2)
    ax.set_ylabel("Fraction of genes (FDR < 0.05)")
    ax.set_title("Significantly imputed genes")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    return fig


def plot_overlap(overlaps: dict[str, dict], a_name="Tangram", b_name="SpaGE"):
    """Stacked bars of shared vs method-specific significant genes, one panel per dataset."""
    fig, axes = plt.subplots(1, len(overlaps), figsize=(7 * len(overlaps), 6), sharey=True, squeeze=False)
    for ax, (ds, o) in zip(axes[0], overlaps.items()):
        shared = [o["shared"], o["shared"]]
        unique = [o["only_a"], o["only_b"]]
        ax.bar([a_name, b_name], shared, color="#7b9ead", edgecolor="black", label="Shared")
        ax.bar([a_name, b_name], unique, bottom=shared, color="#a1d99b", edgecolor="black", label="Unique")
        for c in ax.containers:
            ax.bar_label(c, label_type="center", fmt="%d", weight="bold")
        ax.set_title(f"{ds} (Jaccard = {o['jaccard']:.2f})")
    axes[0, 0].set_ylabel("Significant genes")
    axes[0, 0].legend()
    fig.tight_layout()
    return fig


def plot_property_distributions(props: pd.DataFrame, cols=("sparsity_sp", "moran_i")):
    """Histograms of gene properties, D1 vs D2. ``props`` needs a ``dataset`` column."""
    labels = {"sparsity_sp": "Spatial sparsity", "moran_i": "Moran's I"}
    fig, axes = plt.subplots(1, len(cols), figsize=(7 * len(cols), 5))
    for ax, c in zip(np.atleast_1d(axes), cols):
        sns.histplot(data=props, x=c, hue="dataset", bins=40, stat="density", common_norm=False,
                     palette=DATASET_COLOURS, alpha=0.6, kde=True, ax=ax)
        ax.set_xlabel(labels.get(c, c))
    fig.tight_layout()
    return fig


def plot_property_vs_score(tidy: pd.DataFrame, score: str = "spearman_r",
                           props=("moran_i", "sparsity_sp")):
    """Rows = gene property, columns = dataset; scatter + regression line per method."""
    labels = {"sparsity_sp": "Spatial sparsity", "moran_i": "Moran's I"}
    datasets = sorted(tidy["dataset"].unique())
    fig, axes = plt.subplots(len(props), len(datasets), figsize=(7 * len(datasets), 5 * len(props)),
                             squeeze=False)
    for i, p in enumerate(props):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            for method, colour in METHOD_COLOURS.items():
                d = tidy[(tidy["dataset"] == ds) & (tidy["method"] == method)]
                sns.regplot(data=d, x=p, y=score, ax=ax, color=colour, label=method,
                            scatter_kws={"alpha": 0.25, "s": 15}, line_kws={"lw": 2})
            ax.set(xlabel=labels.get(p, p), ylabel=METRIC_LABELS.get(score, score), title=ds)
            ax.grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend()
    fig.tight_layout()
    return fig


def plot_correlation_bars(corr: pd.DataFrame, group_col: str = "label"):
    """Bar chart of r from ``stats.correlation_table``, one panel per (x, y) pair."""
    pairs = corr[["x", "y"]].drop_duplicates().itertuples(index=False)
    pairs = list(pairs)
    fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 5), squeeze=False)
    for ax, (x, y) in zip(axes[0], pairs):
        d = corr[(corr["x"] == x) & (corr["y"] == y)]
        ax.bar(d[group_col], d["r"], color=[CONDITION_COLOURS.get(g, "grey") for g in d[group_col]],
               edgecolor="black")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylim(-1, 1)
        ax.set_title(f"r({x}, {y})")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def plot_calibration_kde(score_frames: dict[str, pd.DataFrame]):
    """KDE of conformal scores by calibration group, one panel per method."""
    palettes = {"Tangram": "viridis", "SpaGE": "YlOrRd"}
    fig, axes = plt.subplots(1, len(score_frames), figsize=(8 * len(score_frames), 6), sharey=True,
                             squeeze=False)
    for ax, (name, df) in zip(axes[0], score_frames.items()):
        groups = sorted(df["group"].unique())
        pal = sns.color_palette(palettes.get(name, "viridis"), len(groups))
        sns.kdeplot(data=df, x="calibration_score", hue="group", hue_order=groups, palette=pal,
                    common_norm=False, fill=True, alpha=0.1, ax=ax, legend=False)
        ax.legend(handles=[Patch(color=pal[i], label=str(g + 1)) for i, g in enumerate(groups)], title="Group")
        ax.set_title(name)
        ax.set_xlabel("Calibration score")
    fig.tight_layout()
    return fig


def plot_reliability_hist(rel: pd.DataFrame, xlabel: str):
    """Histogram of per-gene error/uncertainty correlation; ``rel`` has columns ``rho`` and ``method``."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data=rel, x="rho", hue="method", bins=40, stat="density", common_norm=False,
                 palette=METHOD_COLOURS, alpha=0.7, kde=True, ax=ax)
    ax.axvline(0, color="black", linestyle="--", lw=2)
    ax.set_xlabel(xlabel)
    return fig


def plot_quartile_boxes(df: pd.DataFrame, method: str, props=("moran_i", "sparsity_sp")):
    labels = {"sparsity_sp": "Spatial sparsity", "moran_i": "Moran's I"}
    pal = sns.color_palette("Blues" if method == "Tangram" else "Reds", 4)
    fig, axes = plt.subplots(1, len(props), figsize=(7 * len(props), 5))
    for ax, p in zip(np.atleast_1d(axes), props):
        sns.boxplot(data=df, x="quartile", y=p, hue="quartile", palette=pal, legend=False, ax=ax)
        ax.set(xlabel="Reliability quartile", ylabel=labels.get(p, p))
    fig.suptitle(method)
    fig.tight_layout()
    return fig


def plot_error_and_pi(adata, pred_key: str, genes, xlim=None, ylim=None, power: float = 0.5):
    """Absolute error and prediction-interval width in space, one row per gene.

    PI width is shown on a ``power`` (sqrt) scale for contrast; colour-bar ticks are
    relabelled on the original scale. ``xlim``/``ylim`` zoom into a region.
    """
    coords = adata.obsm["spatial"]
    mask = np.ones(len(coords), bool)
    if xlim:
        mask &= (coords[:, 0] >= xlim[0]) & (coords[:, 0] <= xlim[1])
    if ylim:
        mask &= (coords[:, 1] >= ylim[0]) & (coords[:, 1] <= ylim[1])
    c = coords[mask]
    fig, axes = plt.subplots(len(genes), 2, figsize=(14, 6 * len(genes)), squeeze=False)
    for row, g in zip(axes, genes):
        err = np.abs(adata.obsm[f"{pred_key}_error"][g].values)[mask]
        pi = (adata.obsm[f"{pred_key}_hi"][g].values - adata.obsm[f"{pred_key}_lo"][g].values).clip(min=0)[mask]
        pi_t = pi ** power
        for ax, vals, title in [(row[0], err, f"|error|: {g}"), (row[1], pi_t, f"PI width: {g}")]:
            im = ax.scatter(c[:, 0], c[:, 1], c=vals, s=5, vmin=vals.min(), vmax=np.percentile(vals, 99),
                            rasterized=True)
            cb = fig.colorbar(im, ax=ax, shrink=0.7)
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
        ticks = cb.get_ticks()
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t ** (1 / power):.2f}" for t in ticks])
    fig.tight_layout()
    return fig


def interactive_actual_vs_predicted(adata, gene: str, pred_key: str = "tangram_predictions"):
    """Plotly figure with linked zoom: measured (top) vs predicted (bottom)."""
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

    x, y = adata.obs[X_COL], adata.obs[Y_COL]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=("Measured", "Predicted"))
    for r, vals in enumerate([adata[:, gene].X, adata.obsm[pred_key][gene].values], start=1):
        fig.add_trace(go.Scattergl(x=x, y=y, mode="markers",
                                   marker=dict(color=display_values(vals), size=2, colorscale="Viridis")),
                      row=r, col=1)
    fig.update_layout(height=900, width=800, showlegend=False, title=f"{gene} ({pred_key})")
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, showticklabels=False)
    return fig
