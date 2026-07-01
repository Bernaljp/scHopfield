"""Import shim so CellOracle 0.20 imports under Python 3.12 + gimmemotifs>=0.18.

CellOracle 0.20 expects `gimmemotifs.motif.default_motifs`, which newer
gimmemotifs renamed. We only need CellOracle's data-loading + perturbation
(Oracle) APIs, not motif scanning, so aliasing the symbol is sufficient.

Usage (in the .venv-co interpreter):
    import analyses._co_shim   # noqa: F401  (must precede `import celloracle`)
    import celloracle as co
"""
import gimmemotifs.motif as _gm

if not hasattr(_gm, "default_motifs"):
    try:
        from gimmemotifs.motif import read_motifs as _read_motifs

        _gm.default_motifs = lambda *a, **k: _read_motifs()
    except Exception:
        _gm.default_motifs = lambda *a, **k: []
