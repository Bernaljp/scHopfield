"""Maximum-likelihood Hill fit: parameter recovery, the bimodality gate and the posterior assignment.

The Hill function x^n / (x^n + k^n) is the CDF of a log-logistic distribution, so samples drawn from
a known log-logistic mixture have a known answer. Draws use the inverse CDF, x = k (u / (1 - u))^(1/n),
with u kept inside [0.005, 0.995]. Untruncated draws put the maximum far out in the heavy right tail,
and the fit's activity threshold (5% of the maximum) would then discard the low component entirely,
which is a property of the threshold on unbounded samples rather than of the likelihood fit.
"""
import numpy as np
import pytest
import anndata as ad

from scHopfield._utils import hill_mle as H
from scHopfield._utils.io import assign_regime, observed_regime, regime_rule


def _draw(k, n, size, rng, lo=0.005):
    u = rng.uniform(lo, 1 - lo, size)
    return k * (u / (1 - u)) ** (1.0 / n)


def _bimodal_sample(rng, m=4000, lo=0.005):
    z = rng.uniform(size=m) < 0.4
    return np.where(z, _draw(2.0, 6.0, m, rng, lo), _draw(10.0, 5.0, m, rng, lo)), z


def test_mixture_recovers_known_parameters_and_passes_the_gate():
    # Untrimmed draws, because trimming the tails makes each component look steeper and biases the
    # recovered exponent upward. The activity threshold is lowered to match: the heavy upper tail then
    # sets a large maximum, and 5% of it would cut into the low component.
    rng = np.random.default_rng(0)
    x, _ = _bimodal_sample(rng, lo=1e-9)
    r = H.fit_hill_mle_gated([x], min_th=1e-4, device="cpu", steps=1500)
    assert r["is_bimodal"][0]
    assert r["k1"][0] == pytest.approx(2.0, rel=0.08)
    assert r["k2"][0] == pytest.approx(10.0, rel=0.08)
    assert r["n1"][0] == pytest.approx(6.0, rel=0.15)
    assert r["n2"][0] == pytest.approx(5.0, rel=0.15)
    assert r["a"][0] == pytest.approx(0.4, abs=0.04)


def test_unimodal_sample_keeps_a_single_hill():
    rng = np.random.default_rng(1)
    x = _draw(3.0, 4.0, 4000, rng)
    r = H.fit_hill_mle_gated([x], device="cpu", steps=1500)
    assert not r["is_bimodal"][0]
    assert r["a"][0] == 1.0
    assert r["k1"][0] == r["k2"][0] and r["n1"][0] == r["n2"][0]
    assert r["k1"][0] == pytest.approx(3.0, rel=0.06)
    assert r["n1"][0] == pytest.approx(4.0, rel=0.1)


def test_single_fit_is_the_maximum_likelihood_one():
    rng = np.random.default_rng(2)
    x = _draw(2.0, 3.0, 3000, rng)
    k, n, _, nll = H.fit_loglogistic_mle([x], np.array([[1.0]]), np.array([[1.5]]), np.ones((1, 1)),
                                         steps=2000, device="cpu")
    best = -np.sum(H.loglogistic_logpdf_np(x, k[0, 0], n[0, 0]))
    assert nll[0] == pytest.approx(best, rel=1e-9)
    for dk, dn in [(1.05, 1.0), (0.95, 1.0), (1.0, 1.05), (1.0, 0.95)]:
        assert -np.sum(H.loglogistic_logpdf_np(x, k[0, 0] * dk, n[0, 0] * dn)) > best


def test_posterior_regime_is_the_argmax_of_the_weighted_densities():
    x = np.linspace(0.05, 30, 500)
    k1, n1, k2, n2, a = 2.0, 6.0, 10.0, 5.0, 0.4
    reg = H.posterior_regime(x, k1, n1, k2, n2, a)
    w1 = a * np.exp(H.loglogistic_logpdf_np(x, k1, n1))
    w2 = (1 - a) * np.exp(H.loglogistic_logpdf_np(x, k2, n2))
    assert np.array_equal(reg, (w2 > w1).astype(np.int8))
    assert not H.posterior_regime(x, k1, n1, k1, n1, 1.0).any()      # single-Hill gene


def test_values_at_or_below_the_activity_threshold_take_component_one():
    # A high-threshold component with the smaller exponent has the larger density as x -> 0, so the
    # untruncated posterior sends non-expressing cells to it; the threshold keeps them in component 1.
    x = np.array([0.0, 0.01, 0.05, 0.1, 5.0, 15.0])
    k1, n1, k2, n2, a = 2.0, 6.0, 10.0, 2.0, 0.5
    raw = H.posterior_regime(x, k1, n1, k2, n2, a)
    assert raw[1] == 1                                                 # the extrapolation artifact
    held = H.posterior_regime(x, k1, n1, k2, n2, a, tau=0.1)
    assert held[:4].tolist() == [0, 0, 0, 0]
    assert held[4:].tolist() == raw[4:].tolist()                        # above tau, unchanged


def test_posterior_assignment_mostly_recovers_the_generating_component():
    rng = np.random.default_rng(3)
    x, z = _bimodal_sample(rng)
    r = H.fit_hill_mle_gated([x], device="cpu")
    reg = H.posterior_regime(x, r["k1"][0], r["n1"][0], r["k2"][0], r["n2"][0], r["a"][0])
    assert np.mean(reg == (~z).astype(np.int8)) > 0.9


def _fitted_object(method):
    import scHopfield as sch
    rng = np.random.default_rng(4)
    x, _ = _bimodal_sample(rng, m=1500)
    y = _draw(3.0, 4.0, 1500, rng)
    a = ad.AnnData(X=np.stack([x, y], 1).astype(np.float32))
    a.layers["Ms"] = a.X.copy()
    sch.pp.fit_all_sigmoids(a, spliced_key="Ms", method=method, device="cpu")
    sch.pp.compute_sigmoid(a, spliced_key="Ms")
    return a


def test_a_likelihood_fit_records_and_uses_the_posterior_rule():
    from scHopfield._utils.math import sigmoid_regime
    a = _fitted_object("mle")
    assert a.uns["scHopfield"]["sigmoid_method"] == "mle"
    assert regime_rule(a) == "posterior"
    assert a.var["sigmoid_mix"].values[0] < 1 and a.var["sigmoid_mix"].values[1] == 1
    genes = np.arange(2)
    R = observed_regime(a, genes, "Ms")
    v = a.var
    expected = sigmoid_regime(np.asarray(a.layers["Ms"], float), v["sigmoid_threshold"].values,
                              v["sigmoid_exponent"].values, v["sigmoid_threshold2"].values,
                              v["sigmoid_exponent2"].values, regime=R)
    assert np.allclose(np.asarray(a.layers["sigmoid"]), expected)
    post = H.posterior_regime(np.asarray(a.layers["Ms"], float)[:, 0], v["sigmoid_threshold"].values[0],
                              v["sigmoid_exponent"].values[0], v["sigmoid_threshold2"].values[0],
                              v["sigmoid_exponent2"].values[0], v["sigmoid_mix"].values[0])
    tau = v["sigmoid_active_min"].values
    assert tau[0] == pytest.approx(0.05 * np.asarray(a.layers["Ms"], float)[:, 0].max())
    X0 = np.asarray(a.layers["Ms"], float)[:, 0]
    post = np.where(X0 <= tau[0], 0, post)
    assert np.array_equal(R[:, 0], post)


def test_an_object_without_the_record_keeps_the_nearest_threshold_rule():
    from scHopfield._utils.math import hill_regime
    a = _fitted_object("mle")
    del a.uns["scHopfield"]["sigmoid_assignment"]
    X = np.asarray(a.layers["Ms"], float)
    R = assign_regime(a, X, np.arange(2))
    v = a.var
    assert np.array_equal(R, hill_regime(X, v["sigmoid_threshold"].values[None, :],
                                         v["sigmoid_threshold2"].values[None, :]))
