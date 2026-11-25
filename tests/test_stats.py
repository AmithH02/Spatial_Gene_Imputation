import numpy as np
import pandas as pd
import pytest
import scipy.stats as st

from spatial_imputation import stats


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    n = 400
    actual = pd.DataFrame(rng.normal(size=(n, 3)), columns=["good", "noise", "const"])
    actual["const"] = 1.0
    pred = pd.DataFrame({
        "good": actual["good"] + rng.normal(0, 0.3, n),
        "noise": rng.normal(size=n),
        "const": rng.normal(size=n),
    })
    return pred, actual


def test_columnwise_pearson_matches_scipy(frames):
    pred, actual = frames
    r = stats.columnwise_pearson(pred[["good", "noise"]].values, actual[["good", "noise"]].values)
    for i, g in enumerate(["good", "noise"]):
        assert r[i] == pytest.approx(st.pearsonr(pred[g], actual[g])[0])


def test_permutation_test(frames):
    pred, actual = frames
    res = stats.permutation_test(pred, actual, n_permutations=199, seed=0)
    assert res.loc["good", "p_value"] == pytest.approx(1 / 200)
    assert bool(res.loc["good", "significant_fdr"])
    assert not bool(res.loc["noise", "significant_fdr"])
    assert np.isnan(res.loc["const", "pearson_r"]) and not res.loc["const", "significant_fdr"]


def test_permutation_test_is_reproducible(frames):
    pred, actual = frames
    a = stats.permutation_test(pred, actual, n_permutations=50, seed=3)
    b = stats.permutation_test(pred, actual, n_permutations=50, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_morans_i_matches_explicit_weights():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(150, 2))
    X = np.column_stack([coords[:, 0] + rng.normal(0, 5, 150), rng.normal(size=150)])
    k = 6
    d = np.linalg.norm(coords[:, None] - coords[None], axis=2)
    np.fill_diagonal(d, np.inf)
    W = np.zeros_like(d)
    for i, nb in enumerate(np.argsort(d, axis=1)[:, :k]):
        W[i, nb] = 1 / k
    Z = X - X.mean(0)
    expected = [(len(z) / W.sum()) * (z @ W @ z) / (z @ z) for z in Z.T]
    np.testing.assert_allclose(stats.morans_i(coords, X, k=k), expected)
    I = stats.morans_i(coords, X, k=k)
    assert I[0] > 0.5 and abs(I[1]) < 0.15


def test_set_overlap():
    o = stats.set_overlap(["a", "b", "c"], ["b", "c", "d", "e"])
    assert o == {"shared": 2, "only_a": 1, "only_b": 2, "jaccard": 2 / 5}


def test_paired_method_test_direction():
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"pearson_r": rng.uniform(0.5, 1, 50), "js": rng.uniform(0, 0.1, 50)})
    b = pd.DataFrame({"pearson_r": a["pearson_r"] - 0.2, "js": a["js"] + 0.2})
    res = stats.paired_method_test(a, b, ["pearson_r", "js"]).set_index("metric")
    assert res.loc["pearson_r", "better"] == "Tangram > SpaGE"
    assert res.loc["js", "better"] == "Tangram > SpaGE"  # lower JS is better
