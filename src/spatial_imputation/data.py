"""Loading, cortex filtering and preprocessing of spatial / scRNA-seq AnnData pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config
from .config import X_COL, Y_COL, CortexMask


# --------------------------------------------------------------------------- loading
def load_raw(key: str):
    """Load the raw spatial section and the scRNA-seq reference for dataset ``key``."""
    import scanpy as sc

    spec = config.get_dataset(key)
    spatial = sc.read_h5ad(config.RAW_DIR / spec.spatial_file)
    reference = sc.read_h5ad(config.RAW_DIR / config.SC_REFERENCE_FILE)
    return spatial, reference


def load_processed(key: str):
    """Load the filtered + normalised (spatial, scRNA-seq) pair written by ``save_processed``."""
    import scanpy as sc

    sp_path, sc_path = config.processed_paths(key)
    return sc.read_h5ad(sp_path), sc.read_h5ad(sc_path)


def save_processed(key: str, spatial, reference) -> None:
    sp_path, sc_path = config.processed_paths(key)
    sp_path.parent.mkdir(parents=True, exist_ok=True)
    spatial.write_h5ad(sp_path)
    reference.write_h5ad(sc_path)


def load_predictions(key: str, method: str, obs_names=None) -> pd.DataFrame:
    """Read a LOO prediction matrix (cells x genes). Optionally re-index to ``obs_names``.

    The CSVs were written with integer row indices, so ``obs_names`` is used to
    align them to the spatial AnnData (same row order).
    """
    preds = pd.read_csv(config.predictions_path(key, method), index_col=0)
    if obs_names is not None:
        if len(obs_names) != len(preds):
            raise ValueError(f"{len(preds)} prediction rows vs {len(obs_names)} cells")
        preds.index = pd.Index(obs_names)
    return preds


def load_analysis_inputs(key: str, methods=config.METHODS):
    """Processed pair with predictions attached as ``obsm['<method>_predictions']``."""
    spatial, reference = load_processed(key)
    for m in methods:
        spatial.obsm[f"{m}_predictions"] = load_predictions(key, m, spatial.obs_names)
    add_spatial_coords(spatial)
    return spatial, reference


# --------------------------------------------------------------------------- filtering
def cortex_mask(x: np.ndarray, y: np.ndarray, mask: CortexMask) -> np.ndarray:
    """Boolean mask of cells outside the ellipse (and outside the optional exclusion box)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x0, y0 = np.median(x) + mask.dx, np.median(y) + mask.dy
    keep = ((x - x0) / mask.a) ** 2 + ((y - y0) / mask.b) ** 2 > 1
    if mask.exclude_box is not None:
        x_min, y_max = mask.exclude_box
        keep &= ~((x > x_min) & (y < y_max))
    return keep


def filter_cortex(spatial, key: str):
    """Return a copy of ``spatial`` restricted to cortex.

    Subcortical cells are dropped because the scRNA-seq reference has low gene
    throughput there, which makes imputation unreliable.
    """
    spec = config.get_dataset(key)
    keep = cortex_mask(spatial.obs[X_COL].values, spatial.obs[Y_COL].values, spec.cortex)
    return spatial[keep].copy()


# --------------------------------------------------------------------------- preprocessing
def lowercase_genes(adata) -> None:
    adata.var_names = pd.Index([g.lower() for g in adata.var_names])
    adata.var_names_make_unique()


def annotate_sparsity(adata) -> None:
    """Add ``var['sparsity']`` = fraction of cells with zero expression."""
    X = adata.X
    nonzero = X.getnnz(axis=0) if sp.issparse(X) else np.count_nonzero(np.asarray(X), axis=0)
    adata.var["sparsity"] = 1 - np.asarray(nonzero).ravel() / adata.n_obs


def preprocess_pair(spatial, reference, normalize: bool = True):
    """Lower-case gene names, log-normalise, subset to shared genes and annotate sparsity.

    Returns new (spatial, reference) AnnData objects restricted to the shared genes.

    Note: ``normalize=True`` runs ``normalize_total`` + ``log1p`` on both objects
    (via TISSUE). Dataset ``d2`` is already log-normalised, so this normalises it a
    second time; that is what the reported results used.
    """
    import tissue.main

    spatial, reference = spatial.copy(), reference.copy()
    lowercase_genes(spatial)
    lowercase_genes(reference)
    tissue.main.preprocess_data(reference, standardize=False, normalize=normalize)
    tissue.main.preprocess_data(spatial, standardize=False, normalize=normalize)

    shared = np.intersect1d(spatial.var_names, reference.var_names)
    spatial = spatial[:, shared].copy()
    reference = reference[:, shared].copy()
    annotate_sparsity(spatial)
    annotate_sparsity(reference)
    return spatial, reference


def add_spatial_coords(adata) -> None:
    """Store ``obs[center_x, center_y]`` as ``obsm['spatial']`` (needed by TISSUE)."""
    adata.obsm["spatial"] = adata.obs[[X_COL, Y_COL]].to_numpy(dtype=float)


def expression_frame(adata, genes=None) -> pd.DataFrame:
    """Dense cells x genes DataFrame of ``adata.X``."""
    sub = adata if genes is None else adata[:, list(genes)]
    X = sub.X.toarray() if sp.issparse(sub.X) else np.asarray(sub.X)
    return pd.DataFrame(X, index=sub.obs_names, columns=sub.var_names)


def sparsity_table(spatial, reference, genes=None) -> pd.DataFrame:
    """Per-gene spatial / single-cell sparsity and their absolute difference."""
    genes = spatial.var_names if genes is None else pd.Index(genes)
    df = pd.DataFrame(
        {
            "sparsity_sp": spatial.var.loc[genes, "sparsity"].values,
            "sparsity_sc": reference.var.loc[genes, "sparsity"].values,
        },
        index=genes,
    )
    df["sparsity_diff"] = (df["sparsity_sp"] - df["sparsity_sc"]).abs()
    return df


def dataset_summary(spatial, reference) -> pd.DataFrame:
    """Cells, genes and transcripts-per-cell for the spatial section and the reference."""

    def counts(adata, use_obs=None):
        if use_obs and use_obs in adata.obs:
            return adata.obs[use_obs].to_numpy(dtype=float)
        return np.asarray(adata.X.sum(axis=1)).ravel()

    rows = {}
    for name, adata, col in [("Spatial", spatial, "transcript_count"), ("Single-cell", reference, None)]:
        c = counts(adata, col)
        rows[name] = {
            "cells": adata.n_obs,
            "genes": adata.n_vars,
            "mean_counts": c.mean(),
            "median_counts": np.median(c),
            "sd_counts": c.std(),
        }
    return pd.DataFrame(rows).T
