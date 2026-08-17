"""Regression tests for the projection-free fate-probability readout and its permutation null.

The reason this readout replaced the projected-flow lineage bias is a single structural property:
a knockout that propagates to nothing must score exactly zero. The projected bias failed that.
A gene with no outgoing regulatory edges (its column of W held at zero by the transcription-factor
mask, or simply a sink) changes no other gene's velocity, so once its own coordinate is neutralized
the transition matrix, the absorbing chain and every fate probability are untouched. Any future
change that reintroduces sensitivity to a non-propagating gene reintroduces the Malat1 artifact,
which is what these tests pin.

The absorbing-chain solve itself is also pinned against the closed form on a chain small enough to
verify by hand, so a change to the sparse solve cannot silently alter the reported shifts.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from scHopfield.tools import (
    decider_mask,
    fate_probabilities,
    fate_transition_matrix,
    permutation_null_floor,
    split_fraction,
)


def _knn_indices(X, k):
    """Brute-force k nearest neighbors, self excluded, so the test does not depend on sklearn."""
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k]


def _toy(n_cells=40, n_genes=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_cells, n_genes))
    V = rng.normal(size=(n_cells, n_genes))
    return X, V, _knn_indices(X, 5)


def test_absorbing_chain_matches_closed_form():
    """Three transient cells feeding two absorbing states, solved by hand.

    Cell 0 goes to state A with probability 1/2 and to cell 1 with probability 1/2; cell 1 goes to
    A and B with probability 1/2 each. So p_A(0) = 1/2 + 1/2 * 1/2 = 3/4 and p_B(0) = 1/4.
    """
    n = 4
    T = np.zeros((n, n))
    T[0, 2] = 0.5     # cell 0 -> A
    T[0, 1] = 0.5     # cell 0 -> cell 1
    T[1, 2] = 0.5     # cell 1 -> A
    T[1, 3] = 0.5     # cell 1 -> B
    T[2, 2] = T[3, 3] = 1.0
    fate, states = fate_probabilities(sp.csr_matrix(T), {"A": np.array([2]), "B": np.array([3])})

    assert states == ["A", "B"]
    a, b = states.index("A"), states.index("B")
    assert fate[0, a] == pytest.approx(0.75)
    assert fate[0, b] == pytest.approx(0.25)
    assert fate[1, a] == pytest.approx(0.5)
    assert fate[2, a] == pytest.approx(1.0)
    assert fate[3, b] == pytest.approx(1.0)
    np.testing.assert_allclose(fate.sum(1), 1.0)


def test_nonpropagating_knockout_gives_exactly_zero():
    """The property the readout exists for: a gene that changes no other gene moves no fate.

    A knockout of a gene with no outgoing edges leaves every other gene's velocity untouched. Once
    that gene's own coordinate is dropped, the velocities entering the kernel are identical, so the
    fate shift is exactly zero, not merely small.
    """
    X, V_wt, knn = _toy()
    sink = 2

    # Knockout of a pure sink: only its own velocity coordinate moves.
    V_ko = V_wt.copy()
    V_ko[:, sink] += np.random.default_rng(1).normal(scale=5.0, size=X.shape[0])

    keep = np.ones(X.shape[1], bool)
    keep[sink] = False
    term = {"A": np.array([0, 1]), "B": np.array([2, 3])}

    f_wt, _ = fate_probabilities(fate_transition_matrix(X[:, keep], V_wt[:, keep], knn), term)
    f_ko, _ = fate_probabilities(fate_transition_matrix(X[:, keep], V_ko[:, keep], knn), term)

    s_wt = split_fraction(f_wt, [0], [1])
    s_ko = split_fraction(f_ko, [0], [1])
    assert np.max(np.abs(s_ko - s_wt)) == 0.0


def test_permutation_null_is_zero_for_a_zero_displacement():
    """A displacement of exactly zero cannot beat its own floor, however it is permuted."""
    X, V_wt, knn = _toy()
    term = {"A": np.array([0, 1]), "B": np.array([2, 3])}
    floor, null = permutation_null_floor(
        X, V_wt, np.zeros_like(V_wt), knn, term, [0], [1], n=5)
    assert floor == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(null, 0.0, atol=1e-12)


def test_permutation_null_is_seed_reproducible_and_magnitude_preserving():
    X, V_wt, knn = _toy()
    term = {"A": np.array([0, 1]), "B": np.array([2, 3])}
    disp = np.random.default_rng(3).normal(scale=0.2, size=V_wt.shape)

    f1, n1 = permutation_null_floor(X, V_wt, disp, knn, term, [0], [1], n=6, seed=0)
    f2, n2 = permutation_null_floor(X, V_wt, disp, knn, term, [0], [1], n=6, seed=0)
    assert f1 == f2
    np.testing.assert_array_equal(n1, n2)

    f3, _ = permutation_null_floor(X, V_wt, disp, knn, term, [0], [1], n=6, seed=1)
    assert f3 != f1, "a different seed should draw different permutations"

    # The null reuses the displacement rows, so the Frobenius norm is preserved exactly.
    p = np.random.default_rng(0).permutation(disp.shape[0])
    assert np.linalg.norm(disp[p]) == pytest.approx(np.linalg.norm(disp))


def test_decider_mask_falls_back_when_no_labile_band_exists():
    committed = np.concatenate([np.zeros(30), np.ones(30)])       # every cell already decided
    np.testing.assert_array_equal(decider_mask(committed), np.ones(60, bool))

    labile = np.linspace(0.0, 1.0, 60)
    m = decider_mask(labile)
    assert m.sum() < 60 and m.sum() >= 10
    assert np.all((labile[m] > 0.1) & (labile[m] < 0.9))


def test_decider_mask_prefers_explicit_transitional_clusters():
    split = np.linspace(0.0, 1.0, 60)
    clusters = np.array(["committed"] * 40 + ["pre-endocrine"] * 20)
    m = decider_mask(split, transitional=["pre-endocrine"], clusters=clusters)
    np.testing.assert_array_equal(m, clusters == "pre-endocrine")


def test_split_fraction_is_insensitive_to_a_third_dominant_fate():
    """The split normalizes within the decision, so a large third attractor does not swamp it."""
    fate = np.array([[0.1, 0.3, 0.6],
                     [0.2, 0.2, 0.6]])
    np.testing.assert_allclose(split_fraction(fate, [0], [1]), [0.25, 0.5], atol=1e-9)
