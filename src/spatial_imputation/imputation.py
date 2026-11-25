"""Leave-one-gene-out (LOO) imputation with Tangram and SpaGE.

For every gene shared by the spatial panel and the scRNA-seq reference, the gene
is removed from the spatial data, the method is trained on the remaining genes,
and the held-out gene is predicted. The result is a cells x genes matrix of
predictions that can be compared with the measured expression.
"""

from __future__ import annotations

import pandas as pd
from tqdm.auto import tqdm

from .config import TANGRAM_CLUSTER_LABEL


def prepare_tangram(spatial, reference) -> None:
    """Run ``tg.pp_adatas`` (stores ``uns['training_genes']`` on both objects, in place)."""
    import tangram as tg

    tg.pp_adatas(reference, spatial)


def tangram_impute(spatial, reference, genes_to_predict, cluster_label=TANGRAM_CLUSTER_LABEL,
                   density_prior="uniform") -> pd.DataFrame:
    """Cluster-mode Tangram: map reference ``cluster_label`` groups onto spatial cells.

    Training genes are ``spatial.uns['training_genes']`` restricted to genes present
    in both objects, so a held-out gene is never used for training.
    """
    import tangram as tg

    train = [g for g in spatial.uns["training_genes"]
             if g in spatial.var_names and g in reference.var_names]
    ad_map = tg.map_cells_to_space(
        reference, spatial,
        cv_train_genes=train,
        mode="clusters",
        cluster_label=cluster_label,
        density_prior=density_prior,
        verbose=False,
    )
    ad_ge = tg.project_genes(ad_map, reference, cluster_label=cluster_label)
    sub = ad_ge[:, list(genes_to_predict)]
    return pd.DataFrame(sub.X, index=sub.obs_names, columns=sub.var_names)


def spage_impute(spatial, reference, genes_to_predict, n_pv: int = 20) -> pd.DataFrame:
    """SpaGE via TISSUE's wrapper (SpaGE standardises internally before computing PVs)."""
    import tissue.main

    return tissue.main.spage_impute(spatial, reference, genes_to_predict=list(genes_to_predict), n_pv=n_pv)


IMPUTERS = {"tangram": tangram_impute, "spage": spage_impute}


def predict_held_out(spatial, reference, gene: str, method: str, **kwargs) -> pd.Series:
    """Predict one gene after masking it from the spatial data."""
    if gene not in spatial.var_names or gene not in reference.var_names:
        raise ValueError(f"Gene '{gene}' must exist in both spatial and reference data.")
    masked = spatial[:, spatial.var_names != gene].copy()
    pred = IMPUTERS[method](masked, reference, [gene], **kwargs)
    return pred[gene].rename(gene)


def loo_impute(spatial, reference, method: str, genes=None, **kwargs) -> pd.DataFrame:
    """Leave-one-gene-out predictions for ``genes`` (default: all shared genes).

    Expects gene names already lower-cased and the pair preprocessed with
    ``data.preprocess_pair``. For Tangram, ``prepare_tangram`` is called first if needed.
    """
    if method not in IMPUTERS:
        raise ValueError(f"Unknown method '{method}'. Options: {sorted(IMPUTERS)}")
    if method == "tangram" and "training_genes" not in spatial.uns:
        prepare_tangram(spatial, reference)

    genes = [g for g in (spatial.var_names if genes is None else genes) if g in reference.var_names]
    preds = [predict_held_out(spatial, reference, g, method, **kwargs)
             for g in tqdm(genes, desc=f"LOO {method}")]
    out = pd.concat(preds, axis=1)
    out.index = spatial.obs_names
    return out
