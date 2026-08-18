"""Regression tests for the perturbation and fate readouts the paper reports.

These readouts used to live beside the figure code, where nothing tested them and nothing stopped
a second copy appearing. Now that they are the package's, these tests pin the properties that make
them the readouts they are, on a synthetic fitted object small enough to reason about:

* a knockout that propagates to nothing moves nothing, exactly, in every readout that claims it;
* clamping several genes at once agrees with clamping one when there is only one;
* evaluating perturbations across processes agrees with evaluating them serially;
* the double-knockout matrix's diagonal is the single-knockout shift, so singles and doubles are
  on one scale and the synergy is a real subtraction rather than a comparison of two conventions;
* the three summaries of the Jacobian pass agree with each other, which is the reason they share
  a pass at all.
"""
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import scHopfield as sch

SINK = 3            # gene index that regulates nothing: its column of W is zero
N_CELLS = 60
N_GENES = 5
CLUSTERS = ["prog", "armA", "armB"]
LINEAGE_PAIRS = [(["armA"], ["armB"], "A", "B")]


@pytest.fixture(scope="module")
def fitted():
    """A small fitted object: three clusters, five genes, one of them a pure sink."""
    rng = np.random.default_rng(0)
    labels = np.array([CLUSTERS[i % 3] for i in range(N_CELLS)])
    expression = rng.uniform(0.1, 2.0, size=(N_CELLS, N_GENES))

    adata = AnnData(X=expression.copy())
    adata.var_names = [f"g{i}" for i in range(N_GENES)]
    adata.obs["clusters"] = pd.Categorical(labels)
    adata.layers["Ms"] = expression.copy()
    adata.obsm["X_umap"] = rng.uniform(-5.0, 5.0, size=(N_CELLS, 2))

    adata.var["scHopfield_used"] = True
    adata.var["sigmoid_threshold"] = rng.uniform(0.4, 1.2, size=N_GENES)
    adata.var["sigmoid_exponent"] = rng.uniform(1.5, 4.0, size=N_GENES)

    w_all = np.zeros((N_GENES, N_GENES))
    for cluster in CLUSTERS:
        W = rng.normal(scale=0.8, size=(N_GENES, N_GENES))
        W[:, SINK] = 0.0                       # regulates nothing, so a knockout cannot propagate
        adata.varp[f"W_{cluster}"] = W
        w_all += np.abs(W)
        adata.var[f"I_{cluster}"] = rng.uniform(0.05, 0.5, size=N_GENES)
        adata.var[f"gamma_{cluster}"] = rng.uniform(0.2, 1.0, size=N_GENES)
    adata.varp["W_all"] = w_all
    adata.uns["scHopfield"] = {"spliced_key": "Ms"}
    return adata


@pytest.fixture(scope="module")
def scaffold(fitted):
    return sch.tl.fate_scaffold(fitted, "clusters", LINEAGE_PAIRS, basis="umap", n_neighbors=10)


def test_multi_gene_clamp_agrees_with_the_single_gene_clamp(fitted):
    """One gene in the mapping must reproduce the dedicated single-gene argument, bit for bit.

    The combinatorial readouts express a joint knockout through the mapping, and the single-gene
    readouts through ``ko_gene``. If those two paths disagreed, the double-knockout synergy would
    be a difference between two conventions rather than a biological quantity.
    """
    _, v_single, _ = sch.tl.model_velocity(fitted, "clusters", ko_gene="g1", ko_level=0.0)
    _, v_clamp, _ = sch.tl.model_velocity(fitted, "clusters", clamp={"g1": 0.0})
    np.testing.assert_array_equal(v_single, v_clamp)

    # and holding two genes is not the same as holding either one, or the fitted object is degenerate
    _, v_joint, _ = sch.tl.model_velocity(fitted, "clusters", clamp={"g1": 0.0, "g2": 0.0})
    assert not np.array_equal(v_joint, v_single)


def test_a_gene_that_regulates_nothing_moves_no_fate(fitted, scaffold):
    """The property every fate readout exists for, checked on the readouts themselves.

    The sink's column of ``W`` is zero, so clamping it changes no other gene's velocity, and its
    own coordinate is neutralized before the kernel is rebuilt. Every fate readout must therefore
    return exactly its wild-type value, not merely something small.
    """
    sink = f"g{SINK}"
    fate = sch.tl.perturbed_fate(fitted, "clusters", scaffold, sink)
    np.testing.assert_array_equal(fate, scaffold["fate_wt"])

    shift = sch.tl.per_cell_fate_shift(fitted, "clusters", LINEAGE_PAIRS, [sink],
                                       scaffold=scaffold)
    assert np.max(np.abs(shift[("A", "B")]["shift"][sink])) == 0.0

    bias = sch.tl.pairwise_fate_bias(fitted, "clusters", LINEAGE_PAIRS, [sink], scaffold=scaffold)
    assert bias[("A", "B")]["bias"][sink] == 0.0

    flow = sch.tl.fate_embedding_flow(fitted, "clusters", LINEAGE_PAIRS, [sink],
                                      scaffold=scaffold)
    assert np.max(np.abs(flow[sink])) == 0.0

    assert sch.tl.regulatory_out_strength(fitted, "clusters")[sink] == 0.0


def test_a_regulator_does_move_fate(fitted, scaffold):
    """The companion to the zero above: the construction is not simply insensitive to everything."""
    moved = [g for g in ("g0", "g1", "g2", "g4")
             if np.max(np.abs(sch.tl.perturbed_fate(fitted, "clusters", scaffold, g)
                              - scaffold["fate_wt"])) > 0]
    assert moved, "no regulator moved fate at all; the fixture is degenerate"


def test_parallel_evaluation_agrees_with_serial(fitted, scaffold):
    """Forking must not change a number, only how long it takes to get it."""
    tasks = [["g0"], ["g1"], ["g0", "g1"], ["g2"]]
    serial = sch.tl.perturbed_fates(fitted, "clusters", scaffold, tasks, workers=1)
    parallel = sch.tl.perturbed_fates(fitted, "clusters", scaffold, tasks, workers=3)
    assert len(serial) == len(parallel) == len(tasks)
    for a, b in zip(serial, parallel):
        np.testing.assert_array_equal(a, b)


def test_dose_zero_reproduces_the_knockout(fitted, scaffold):
    """Panel e is meant to be the dose-zero slice of panel f, so the two must agree exactly."""
    genes = ["g0", "g1"]
    dose = sch.tl.dose_fate_bias(fitted, "clusters", LINEAGE_PAIRS, genes, fractions=[0.0, 1.0],
                                 scaffold=scaffold)
    bias = sch.tl.pairwise_fate_bias(fitted, "clusters", LINEAGE_PAIRS, genes, scaffold=scaffold)
    for gene in genes:
        at_zero = dose[("A", "B")][gene].query("level_frac == 0.0")["fate_bias"].iloc[0]
        assert at_zero == pytest.approx(bias[("A", "B")]["bias"][gene], abs=1e-12)


def test_double_knockout_matrix_is_consistent_with_its_singles(fitted, scaffold):
    """Diagonal is the single knockout, the matrices are symmetric, synergy is the subtraction."""
    genes = ["g0", "g1", "g2"]
    axes = sch.tl.lineage_pair_axes(scaffold, LINEAGE_PAIRS)
    blocks, _, _ = sch.tl.double_knockout_matrix(fitted, "clusters", scaffold, genes, axes)
    block = blocks[("A", "B")]

    for i, gene in enumerate(block["genes"]):
        assert block["matrix"][i, i] == pytest.approx(block["single"][gene], abs=1e-12)

    np.testing.assert_allclose(block["matrix"], block["matrix"].T, equal_nan=True)
    np.testing.assert_allclose(block["synergy_matrix"], block["synergy_matrix"].T, equal_nan=True)

    idx = {g: i for i, g in enumerate(block["genes"])}
    for (g1, g2), synergy in block["synergy"].items():
        joint = block["matrix"][idx[g1], idx[g2]]
        assert synergy == pytest.approx(joint - (block["single"][g1] + block["single"][g2]),
                                        abs=1e-12)


def test_the_jacobian_summaries_agree_with_each_other(fitted):
    """The three entry points share one finite-difference pass, so they must not disagree."""
    genes = ["g0", "g1"]
    groups = {"A vs B": genes}
    full = sch.tl.jacobian_knockout_response(fitted, "clusters", genes=genes,
                                             lineage_pairs=LINEAGE_PAIRS, groups=groups)
    narrow = sch.tl.jacobian_response(fitted, "clusters", genes)
    for gene in genes:
        pd.testing.assert_series_equal(full["response"][gene], narrow[gene])
        assert gene not in full["response"][gene].index      # a gene is not its own target

    push = sch.tl.jacobian_commitment_push(fitted, "clusters", LINEAGE_PAIRS, groups)
    for gene in genes:
        np.testing.assert_array_equal(push[gene][2], full["commitment_push"][gene][2])
        assert push[gene][0] == "A" and push[gene][1] == "B"


def test_regulatory_coupling_is_a_normalized_symmetric_overlap(fitted):
    genes = ["g0", "g1", "g2", f"g{SINK}"]
    coupling = sch.tl.regulatory_coupling(fitted, genes)
    assert list(coupling.index) == genes and list(coupling.columns) == genes
    np.testing.assert_allclose(coupling.values, coupling.values.T)
    assert coupling.values.min() >= 0.0 and coupling.values.max() <= 1.0 + 1e-12
    for gene in ("g0", "g1", "g2"):
        assert coupling.loc[gene, gene] == pytest.approx(1.0)
    # the sink has an all-zero out-profile, so it couples to nothing, itself included
    assert coupling.loc[f"g{SINK}"].abs().max() == 0.0


def test_commitment_time_is_defined_and_zero_for_a_sink(fitted, scaffold):
    result = sch.tl.commitment_time(fitted, "clusters", LINEAGE_PAIRS,
                                    ["g0", f"g{SINK}"], scaffold=scaffold)
    assert set(result) == {"g0", f"g{SINK}"}
    assert result[f"g{SINK}"] == 0.0
    assert np.isfinite(result["g0"])
