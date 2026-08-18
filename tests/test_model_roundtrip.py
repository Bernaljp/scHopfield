"""A saved model must come back as the model that was saved.

Two defects found while writing the tutorials, both reachable by a reader following the
documented idiom, both fixed here and pinned below.

``save_model`` wrote only the first Hill component, while the canonical ``bimodal=True``
fit produces two. ``compute_sigmoid`` branches on whether the second component is present
and, when it is not, falls through to the ordinary single Hill. So save, load, recompute,
which is the round trip the docstring shows, returned an activation the model was never fit
with, and every energy, velocity and Jacobian built on it was wrong without saying so. On
pancreas at 2,000 genes that was 629 genes with a real second component, 7.47% of the
activation entries differing, and a correlation of 0.933 where it should have been 1.

``BaseCircuit.simulate`` resolved its default initial state through an
``initial_conditions`` attribute that neither shipped circuit defines, so the default path
raised ``AttributeError`` for every public caller.
"""
import warnings

import numpy as np
import pytest
from anndata import AnnData

import scHopfield as sch
from scHopfield.validation.circuits import OscillatorCircuit, ToggleCircuit
from scHopfield.validation.circuits.base import BaseCircuit

BIMODAL_COLS = ("sigmoid_mix", "sigmoid_threshold2", "sigmoid_exponent2")


def _two_regime_adata(n_cells=240, n_genes=12, seed=0):
    """Half the genes get two well-separated expression modes, so the bimodal fit is
    genuinely accepted rather than falling back to a single Hill."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_cells, n_genes))
    half = n_cells // 2
    for g in range(n_genes):
        if g % 2 == 0:
            x[:, g] = np.concatenate([
                np.abs(rng.normal(3.0, 0.30, half)),
                np.abs(rng.normal(12.0, 0.80, n_cells - half)),
            ])
        else:
            x[:, g] = rng.gamma(3.0, 0.8, n_cells)
    a = AnnData(X=x.copy())
    a.layers["Ms"] = x.copy()
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs["cluster"] = ["c0"] * n_cells
    return a


@pytest.fixture(scope="module")
def fitted():
    a = _two_regime_adata()
    sch.pp.fit_all_sigmoids(a, spliced_key="Ms", bimodal=True)
    assert (a.var["sigmoid_mix"].values < 1 - 1e-9).any(), \
        "fixture is not exercising the bimodal branch"
    sch.pp.compute_sigmoid(a, spliced_key="Ms")
    # save_model requires a fitted network; the round trip under test is the var columns.
    a.varp["W_c0"] = np.zeros((a.n_vars, a.n_vars))
    a.var["I_c0"] = 0.0
    a.var["gamma_c0"] = 1.0
    return a


def _reload(fitted, tmp_path, drop=(), keep_uns=True):
    """Save ``fitted``, optionally drop columns from the file's var group to stand in for
    a checkpoint written before the fix, and load it into a fresh AnnData. ``keep_uns``
    False also drops the recorded fit mode, as a genuinely pre-fix file would."""
    import json

    import h5py

    path = tmp_path / "model.h5sch"
    sch.tl.save_model(fitted, str(path), overwrite=True)
    if drop or not keep_uns:
        with h5py.File(path, "a") as f:
            for key in drop:
                if key in f["var"]:
                    del f["var"][key]
            if not keep_uns:
                meta = json.loads(f.attrs["uns_scHopfield"])
                meta.pop("sigmoid_bimodal", None)
                f.attrs["uns_scHopfield"] = json.dumps(meta)
    b = _two_regime_adata()
    sch.tl.load_model(b, str(path))
    return b


# --------------------------------------------------------------------------------------
# the second Hill component survives the round trip
# --------------------------------------------------------------------------------------
def test_fit_writes_the_second_component_in_both_modes():
    for bimodal in (True, False):
        a = _two_regime_adata(n_cells=120, n_genes=4)
        sch.pp.fit_all_sigmoids(a, spliced_key="Ms", bimodal=bimodal)
        for col in BIMODAL_COLS:
            assert col in a.var, f"{col} missing after bimodal={bimodal}"
        assert a.uns["scHopfield"]["sigmoid_bimodal"] is bimodal


def test_save_model_persists_the_second_component(fitted, tmp_path):
    b = _reload(fitted, tmp_path)
    for col in BIMODAL_COLS:
        assert col in b.var, f"{col} did not survive save_model/load_model"
        np.testing.assert_array_equal(b.var[col].values, fitted.var[col].values)


def test_round_trip_reproduces_the_activation_exactly(fitted, tmp_path):
    b = _reload(fitted, tmp_path)
    sch.pp.compute_sigmoid(b, spliced_key="Ms")
    np.testing.assert_array_equal(b.layers["sigmoid"], fitted.layers["sigmoid"])


# --------------------------------------------------------------------------------------
# a checkpoint written before the fix degrades loudly, not silently
# --------------------------------------------------------------------------------------
def test_load_model_says_so_when_the_file_has_only_the_first_component(fitted, tmp_path):
    """Only the file shows the loss, so the warning belongs at load time."""
    with pytest.warns(UserWarning, match="first Hill component"):
        b = _reload(fitted, tmp_path, drop=BIMODAL_COLS, keep_uns=False)
    # it still loads, and still yields the single-Hill activation, just not quietly
    sch.pp.compute_sigmoid(b, spliced_key="Ms")
    assert not np.array_equal(b.layers["sigmoid"], fitted.layers["sigmoid"])


def test_compute_sigmoid_refuses_when_the_object_records_a_bimodal_fit(fitted, tmp_path):
    with pytest.warns(UserWarning):
        b = _reload(fitted, tmp_path, drop=BIMODAL_COLS)
    assert b.uns["scHopfield"]["sigmoid_bimodal"] is True
    with pytest.raises(ValueError) as e:
        sch.pp.compute_sigmoid(b, spliced_key="Ms")
    assert "fit_all_sigmoids" in str(e.value)     # how to get it back


def test_hand_built_single_hill_parameters_are_not_second_guessed():
    """A synthetic circuit with a known Hill sets threshold and exponent by hand and is
    legitimately single component. The guard must not fire on it: tutorial 02 does exactly
    this, four times."""
    x = np.abs(np.random.default_rng(0).normal(1.5, 0.5, (40, 2)))
    a = AnnData(X=x.copy())
    a.layers["Ms"] = x.copy()
    a.var_names = ["x1", "x2"]
    a.var["scHopfield_used"] = True
    a.var["sigmoid_threshold"] = 1.0
    a.var["sigmoid_exponent"] = 4.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sch.pp.compute_sigmoid(a, spliced_key="Ms")
    assert a.layers["sigmoid"].shape == x.shape


def test_a_complete_round_trip_is_silent(fitted, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b = _reload(fitted, tmp_path)
        sch.pp.compute_sigmoid(b, spliced_key="Ms")


# --------------------------------------------------------------------------------------
# simulate() without an initial state
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("circuit", [ToggleCircuit, OscillatorCircuit])
def test_simulate_says_what_to_pass_instead_of_raising_attributeerror(circuit):
    with pytest.raises(TypeError) as e:
        circuit().simulate(t_end=1.0, n_samples=5)
    msg = str(e.value)
    assert circuit.__name__ in msg                    # which circuit
    assert "sample_initial_conditions" in msg         # how to get an initial state
    assert "initial_state" in msg                     # what to pass


@pytest.mark.parametrize("circuit", [ToggleCircuit, OscillatorCircuit])
def test_the_remedy_the_message_names_actually_runs(circuit):
    c = circuit()
    x0 = c.sample_initial_conditions(1)[0]
    t, x = c.simulate(t_end=5.0, n_samples=11, initial_state=x0)
    assert t.shape == (11,)
    assert x.shape == (11, len(c.state_names))
    np.testing.assert_allclose(x[0], x0, atol=1e-6)


def test_a_circuit_that_defines_initial_conditions_still_needs_no_argument():
    """The biophysical circuits kept out of the package do define the attribute, and the
    default path stays available to them."""
    class Decay(BaseCircuit):
        initial_conditions = {"A": 2.0, "B": 5.0}

        @property
        def state_names(self):
            return ("A", "B")

        def rhs(self, x):
            return -x

    t, x = Decay().simulate(t_end=1.0, n_samples=3)
    np.testing.assert_allclose(x[0], [2.0, 5.0], atol=1e-6)
