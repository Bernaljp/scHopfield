"""The figure scripts must fail loudly rather than draw something wrong.

These pin the contract, not the pixels. Three silent-degradation shapes were found in the
shipping figure code, and each one drew a panel that a reader would take as a result:

- a per-dataset registry whose miss returned an empty default, so a ``--dataset`` the panel
  was never curated for drew a zero percent enrichment bar under a caption asserting
  enrichment;
- a first-existing-path fallback that collapsed the two arms of a comparison onto one file,
  so the improvement panels drew a flat zero that reads as a negative result;
- an input that was tested with ``os.path.exists`` and skipped, so the panel came out blank.

The guards are in ``reproducibility/guards.py`` and the call sites are checked here by
source inspection rather than by rendering, because rendering needs the report tree.
"""
import os
import pathlib
import re
import sys

import pytest

REPRO = pathlib.Path(__file__).resolve().parents[1] / "reproducibility"
sys.path.insert(0, str(REPRO))

import guards                                                    # noqa: E402


# --------------------------------------------------------------------------------------
# require_file
# --------------------------------------------------------------------------------------
def test_require_file_returns_the_path_when_it_is_there(tmp_path):
    p = tmp_path / "driver_scores_1.csv"
    p.write_text("gene,score\n")
    assert guards.require_file(str(p), "panel b", "run the report") == str(p)


def test_require_file_names_the_path_and_the_producer(tmp_path):
    missing = str(tmp_path / "driver_scores_1.csv")
    with pytest.raises(FileNotFoundError) as e:
        guards.require_file(missing, "panel b, the driver scores", "run the pancreas report")
    msg = str(e.value)
    assert missing in msg                       # which file
    assert "panel b" in msg                     # which panel goes missing without it
    assert "run the pancreas report" in msg     # how to get it


# --------------------------------------------------------------------------------------
# require_distinct
# --------------------------------------------------------------------------------------
def test_require_distinct_passes_when_the_two_arms_are_two_files(tmp_path):
    a = tmp_path / "adata_analyzed_bimodal.h5ad"; a.write_text("a")
    b = tmp_path / "adata_analyzed_singlehill.h5ad"; b.write_text("b")
    guards.require_distinct(str(a), str(b), "panels c and h", "re-fit")


def test_require_distinct_stops_when_a_fallback_collapsed_them(tmp_path):
    one = tmp_path / "adata_analyzed.h5ad"; one.write_text("a")
    with pytest.raises(FileNotFoundError) as e:
        guards.require_distinct(str(one), str(one), "panels c and h, the two-component "
                                "versus single-Hill comparison", "flip config.BIMODAL_HILL")
    msg = str(e.value)
    assert "itself" in msg                      # says what the comparison actually did
    assert "flip config.BIMODAL_HILL" in msg


def test_require_distinct_sees_through_a_symlink(tmp_path):
    """Two paths, one file. The comparison is still a file against itself."""
    real = tmp_path / "adata_analyzed.h5ad"; real.write_text("a")
    link = tmp_path / "adata_analyzed_singlehill.h5ad"
    link.symlink_to(real)
    with pytest.raises(FileNotFoundError):
        guards.require_distinct(str(real), str(link), "panels c and h", "re-fit")


# --------------------------------------------------------------------------------------
# require_dataset_entry
# --------------------------------------------------------------------------------------
def test_require_dataset_entry_returns_the_curated_entry():
    reg = {"pancreas": ["Neurog3", "Pax4"]}
    assert guards.require_dataset_entry(reg, "pancreas", "R", "panel g") == ["Neurog3", "Pax4"]


def test_require_dataset_entry_lists_what_the_registry_does_cover():
    reg = {"pancreas": ["Neurog3"], "paul15": ["Gata1"]}
    with pytest.raises(ValueError) as e:
        guards.require_dataset_entry(reg, "schwann", "LINEAGE_BY_DATASET", "panel g")
    msg = str(e.value)
    assert "schwann" in msg                     # which dataset was asked for
    assert "pancreas, paul15" in msg            # which ones would work
    assert "LINEAGE_BY_DATASET" in msg          # where to add it


def test_require_dataset_entry_treats_an_empty_entry_as_no_entry():
    with pytest.raises(ValueError):
        guards.require_dataset_entry({"schwann": []}, "schwann", "R", "panel g")


def test_the_message_survives_the_traceback():
    """KeyError renders its argument through repr, which turns a multi-line explanation
    into one line of escaped newlines. The guard must not use it."""
    with pytest.raises(Exception) as e:
        guards.require_dataset_entry({"pancreas": [1]}, "schwann", "R", "panel g")
    assert not isinstance(e.value, KeyError)
    assert "\\n" not in str(e.value) and "\n" in str(e.value)


# --------------------------------------------------------------------------------------
# warn_once
# --------------------------------------------------------------------------------------
def test_warn_once_says_it_once_per_process(capsys):
    guards._WARNED.discard("k")
    guards.warn_once("k", "drawn with matplotlib rather than TikZ")
    guards.warn_once("k", "drawn with matplotlib rather than TikZ")
    err = capsys.readouterr().err
    assert err.count("drawn with matplotlib rather than TikZ") == 1
    assert err.startswith("WARNING:")


# --------------------------------------------------------------------------------------
# The call sites, read from source: no rendering, so no report tree needed.
# --------------------------------------------------------------------------------------
def _src(name):
    return (REPRO / name).read_text()


def test_extended_data_fig2_has_no_module_wide_pancreas_lineage():
    """It was a module global used unconditionally by three panels, on a script that takes
    --dataset. Now a registry, resolved once, that stops on a dataset it does not cover."""
    s = _src("make_sigmoid_activation.py")
    assert "PANCREAS_LINEAGE" not in s
    assert "LINEAGE_BY_DATASET" in s
    assert "require_dataset_entry" in s


def test_extended_data_fig2_guards_the_two_arms_of_its_comparison():
    s = _src("make_sigmoid_activation.py")
    assert "require_distinct" in s
    # the fallback itself may stay; what may not stay is reading both arms unguarded
    assert s.index("require_distinct") < s.index("ad.read_h5ad(p_bi)")


def test_figure1_takes_the_prior_it_was_fit_with_and_keeps_no_false_fallback():
    """`in_prior = set(EDGES)` marked every drawn edge as in the prior, on the one panel
    whose subject is scaffold membership, and the file it fell back from stopped existing
    once the prior became a fetched table."""
    s = _src("make_framework_figure.py")
    assert "in_prior = set(EDGES)" not in s
    assert "fetch_base_grn" in s
    assert "SCAFFOLD.get(" not in s


def test_figure4_docstring_names_the_figure_it_writes():
    s = _src("make_energy_jacobian.py")
    head = s[:s.index('"""', 3)]
    assert "Figure 4" in head and "figure 3" not in head
    assert 'Figure4.pdf' in s


def test_figure4_circuits_announce_a_drawn_fallback():
    """The poster path has drawn its circuits with matplotlib since the promotion, but it
    did so without a word, so a run on a machine with no TeX looked like a normal run."""
    s = _src("make_energy_jacobian.py")
    assert "jacobian-network-fallback" in s
    assert "_draw_mini_network" in s


def test_figure4_circuits_are_type_checked():
    """The save() skipped the journal's 5 pt floor for the whole figure, on the note that
    the circuits are rasters. True of the typeset circuits, not of the drawn fallback, which
    put 48 gene symbols at 4.2 pt on the page precisely when nobody was checking."""
    s = _src("make_energy_jacobian.py")
    assert "check=False" not in s
    assert "fontsize=TYPE_FLOOR" in s


def test_figure5_panel_a_falls_back_in_both_renderers():
    """One resolver, called twice. The poster renderer used to have no fallback at all and
    simply left the velocity panel blank."""
    s = _src("make_perturbation_dynamics.py")
    assert s.count("draw_panel_a(ax_a, ds, C, emb, clusters, colors)") == 2
    assert "panel-a-render:" in s


def test_figure5_driver_scores_are_required_not_optional():
    s = _src("make_perturbation_dynamics.py")
    assert "if os.path.exists(csvp)" not in s
    assert s.count("driver_scores_") == 2
    assert s.count("require_file(") == 2


def test_a_named_pseudotime_column_is_not_silently_recomputed():
    """rutils computed a fresh diffusion pseudotime when the configured column was absent,
    which fits different dynamics from the same input and reports success."""
    s = _src("rutils.py")
    assert 'cfg.get("pseudotime_key", "Pseudotime")' not in s
    assert "would fit different dynamics" in s


# Figure 1 is not in this list on purpose: its fix was to delete a fallback, not to add a
# guard, so it needs nothing from this module.
@pytest.mark.parametrize("script", [
    "make_sigmoid_activation.py", "make_energy_jacobian.py",
    "make_perturbation_dynamics.py", "make_cross_dataset.py", "rutils.py",
])
def test_every_guarded_script_imports_the_shared_guards(script):
    assert re.search(r"^import guards", _src(script), re.M)


def test_house_style_no_em_dashes_in_the_guard_module():
    assert "—" not in guards.__doc__
    assert "—" not in (REPRO / "guards.py").read_text()
