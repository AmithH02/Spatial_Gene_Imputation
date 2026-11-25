import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from spatial_imputation import data
from spatial_imputation.config import CortexMask


def test_cortex_mask_keeps_outside_ellipse():
    x = np.array([0.0, 0.0, 10.0, -10.0, 0.0])
    y = np.array([0.0, 1.0, 0.0, 0.0, 10.0])
    keep = data.cortex_mask(x, y, CortexMask(dx=0, dy=0, a=5, b=5))
    np.testing.assert_array_equal(keep, [False, False, True, True, True])


def test_cortex_mask_exclude_box():
    x = np.array([10.0, -10.0, 0.0])
    y = np.array([0.0, 0.0, 10.0])
    keep = data.cortex_mask(x, y, CortexMask(dx=0, dy=0, a=5, b=5, exclude_box=(5, 5)))
    np.testing.assert_array_equal(keep, [False, True, True])


@pytest.mark.parametrize("sparse", [False, True])
def test_annotate_sparsity(sparse):
    X = np.array([[0, 1], [0, 2], [3, 0], [0, 0]], dtype=float)
    a = ad.AnnData(sp.csr_matrix(X) if sparse else X)
    data.annotate_sparsity(a)
    np.testing.assert_allclose(a.var["sparsity"].values, [0.75, 0.5])


def test_sparsity_table():
    a = ad.AnnData(np.zeros((2, 2)), var=pd.DataFrame({"sparsity": [0.2, 0.9]}, index=["g1", "g2"]))
    b = ad.AnnData(np.zeros((2, 2)), var=pd.DataFrame({"sparsity": [0.5, 0.4]}, index=["g1", "g2"]))
    t = data.sparsity_table(a, b)
    np.testing.assert_allclose(t["sparsity_diff"].values, [0.3, 0.5])


def test_counts_per_cell_prefers_obs_counts():
    a = ad.AnnData(np.log1p(np.ones((3, 2))), obs=pd.DataFrame({"n_counts": [10.0, 20.0, 30.0]}))
    np.testing.assert_allclose(data.counts_per_cell(a), [10, 20, 30])
    b = ad.AnnData(np.ones((2, 3)))
    np.testing.assert_allclose(data.counts_per_cell(b), [3, 3])
