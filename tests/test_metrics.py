import numpy as np
import pandas as pd
import pytest
import scipy.stats as st
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import auc

from spatial_imputation import metrics


@pytest.fixture
def xy():
    rng = np.random.default_rng(1)
    x = rng.gamma(1.0, 1.0, 500)
    y = x + rng.normal(0, 0.5, 500).clip(0)
    return x, y


def test_js_matches_scipy(xy):
    x, y = xy
    assert metrics.js_divergence(x, y) == pytest.approx(jensenshannon(x, y) ** 2)


def test_js_does_not_modify_inputs(xy):
    x, y = xy
    x0 = x.copy()
    metrics.js_divergence(x, y)
    np.testing.assert_array_equal(x, x0)


def test_js_zero_vector_is_nan():
    assert np.isnan(metrics.js_divergence(np.zeros(5), np.ones(5)))


def test_ssim_matches_original_formula(xy):
    x, y = xy
    xs, ys = x / x.max(), y / y.max()
    num = (2 * xs.mean() * ys.mean() + 0.01) * (2 * np.cov(xs, ys)[0, 1] + 0.03)
    den = (xs.mean() ** 2 + ys.mean() ** 2 + 0.01) * (np.var(xs) + np.var(ys) + 0.03)
    assert metrics.ssim_vector(x, y) == pytest.approx(num / den)
    assert metrics.ssim_vector(x, x) == pytest.approx(1, abs=1e-3)


def test_gene_metrics_columns_and_values(xy):
    x, y = xy
    pred = pd.DataFrame({"a": x, "b": y[::-1]})
    act = pd.DataFrame({"a": y, "b": y, "c": y})
    m = metrics.gene_metrics(pred, act)
    assert list(m.index) == ["a", "b"]
    assert m.loc["a", "pearson_r"] == pytest.approx(st.pearsonr(x, y)[0])
    assert m.loc["a", "RMSE"] == pytest.approx(np.sqrt(np.mean((x - y) ** 2)))


def _original_eval_auc(scores, sparsity):
    """Verbatim logic of eval_metric() from the original notebooks."""
    xs, ys = list(sparsity), list(scores)
    pol = np.poly1d(np.polyfit(xs, ys, 2))
    pol_xs = np.linspace(0, 1, 10)
    pol_ys = [min(pol(x), 1) for x in pol_xs]
    root = None
    for r in pol.r:
        if np.isreal(r) and 0 <= r <= 1:
            root = np.real(r)
            break
    if root is not None:
        pol_xs = np.append(pol_xs, root)
        pol_ys = np.append(pol_ys, 0)
    pol_xs = np.append(pol_xs, 1)
    pol_ys = np.append(pol_ys, pol(1))
    filt = [(x, y) for x, y in zip(pol_xs, pol_ys) if 0 <= x <= 1 and 0 <= y <= 1]
    pol_xs, pol_ys = zip(*sorted(filt))
    return np.real(auc(pol_xs, pol_ys))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_auc_matches_original(seed):
    rng = np.random.default_rng(seed)
    sp = rng.uniform(0, 1, 200)
    sc = np.clip(0.8 - 0.7 * sp + rng.normal(0, 0.1, 200), 0, 1)
    assert metrics.score_sparsity_auc(sc, sp)[0] == pytest.approx(_original_eval_auc(sc, sp))
