"""Fetch a CellOracle base gene regulatory network without redistributing one.

A *base GRN* is the prior-knowledge table :func:`build_scaffold` turns into a
scaffold: one row per target gene, one binary column per transcription factor. The
three tables scHopfield was developed against are built and distributed by
CellOracle, and this module downloads them on demand instead of shipping a copy.

Why fetch rather than ship
--------------------------
CellOracle's ``LICENSE`` is a modified Apache License 2.0 whose header restricts use
to non-commercial academic purposes, while scHopfield is MIT. Committing one of these
tables into an MIT repository would offer it onward under terms we do not hold, so
scHopfield distributes none of them. Downloading a file from its publisher is not
redistribution, which is why this route exists at all. **The tables are not covered by
scHopfield's license**; see ``DATA_SOURCES.md`` for the restriction, the citations, and
the manual alternatives.

Each table is addressed by a pinned commit rather than by ``master``, so a
reorganization upstream cannot silently change what is fetched, and every download is
checked against a recorded sha256 before it is cached or returned. The registry that
records those pins is ``BASE_GRNS``; the sha256 values and the pinned commits are also
written out in ``DATA_SOURCES.md`` so they can be read without importing anything.

Three tables, addressed by name
-------------------------------
``mouse_atlas``
    The mouse sci-ATAC-seq atlas base GRN (mm9, ``v202204``). The mouse prior used
    throughout the paper. Aliased as ``mouse``.
``human_promoter``
    The human promoter base GRN (hg19, ``fpr2``). The only human table. Aliased as
    ``human``.
``mouse_promoter``
    The mouse promoter base GRN (mm10, ``fpr2``). The second arm of the
    driver-stability analysis, which compares a fit under the atlas prior against one
    under the promoter prior.

The genome and threshold in each description are load-bearing. The human table is
hg19, not hg38, and both promoter tables are ``fpr2``, not ``fpr1``; a different
choice gives a different scaffold and different fitted parameters.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from typing import Optional

import pandas as pd

__all__ = ["fetch_base_grn"]

#: Raw-content host for the CellOracle repository. A commit is appended to it, never
#: a branch name, so the bytes behind a URL cannot change under us.
_CELLORACLE_RAW = "https://raw.githubusercontent.com/morris-lab/CellOracle"

#: Seconds to wait on the download before giving up. Long enough for a 9 MB file on a
#: slow connection, short enough that a hung transfer fails rather than hangs.
_TIMEOUT = 120

#: The pinned tables. ``loader`` names the CellOracle function that fetches the same
#: file, quoted in error messages so the manual fallback is always one call away.
BASE_GRNS = {
    "mouse_atlas": dict(
        description="mouse sci-ATAC-seq atlas base GRN (mm9, v202204)",
        commit="77cd39fdf77ec931d55437b8b728830bf0b38ee5",
        path=(
            "celloracle/data/TFinfo_data/mm9_mouse_atac_atlas_data_TSS_and_cicero"
            "_0.9_accum_threshold_10.5_DF_peaks_by_TFs_v202204.parquet"
        ),
        sha256="adb95f68f3b03c8522eb9cf55a0a4235bc663c991a10dc655902c1f1cd854129",
        n_bytes=9_448_146,
        loader="load_mouse_scATAC_atlas_base_GRN()",
    ),
    "human_promoter": dict(
        description="human promoter base GRN (hg19, gimmemotifs v5, fpr2)",
        commit="22b29abc469a361ce14e6249ca1413bb91e327fc",
        path=(
            "celloracle/data/promoter_base_GRN/hg19_TFinfo_dataframe"
            "_gimmemotifsv5_fpr2_threshold_10_20210630.parquet"
        ),
        sha256="2be7c71cc37906805e8fdc8a717f4d31406e1b0b566cbfcc1564acddbc0caa8c",
        n_bytes=5_520_100,
        loader='load_human_promoter_base_GRN(version="hg19_gimmemotifsv5_fpr2")',
    ),
    "mouse_promoter": dict(
        description="mouse promoter base GRN (mm10, gimmemotifs v5, fpr2)",
        commit="22b29abc469a361ce14e6249ca1413bb91e327fc",
        path=(
            "celloracle/data/promoter_base_GRN/mm10_TFinfo_dataframe"
            "_gimmemotifsv5_fpr2_threshold_10_20210630.parquet"
        ),
        sha256="32de319419199f53bbdbf7c8545394a70052c7d14918d59b54c96593000dea40",
        n_bytes=4_633_084,
        loader='load_mouse_promoter_base_GRN(version="mm10_gimmemotifsv5_fpr2")',
    ),
}

#: Species names accepted in place of a table name, resolving to the table that
#: species was fit with in the paper. ``mouse`` is ambiguous on its own, since two
#: mouse tables exist; it resolves to the atlas because that is the mouse prior behind
#: every mouse figure.
_ALIASES = {"mouse": "mouse_atlas", "human": "human_promoter"}


def _resolve(name: str) -> str:
    """Return the ``BASE_GRNS`` key for a table name or species alias."""
    key = _ALIASES.get(str(name).lower(), str(name).lower())
    if key not in BASE_GRNS:
        raise ValueError(
            f"unknown base GRN {name!r}. Available tables: "
            f"{', '.join(sorted(BASE_GRNS))}. "
            f"Species aliases: {', '.join(f'{a} -> {t}' for a, t in sorted(_ALIASES.items()))}."
        )
    return key


def _cache_root(cache_dir: Optional[str]) -> str:
    """Return the directory downloads are cached in.

    An explicit ``cache_dir`` wins; otherwise ``SCHOPFIELD_CACHE`` if it is set, which
    is the way to relocate the cache on a machine with no writable home directory;
    otherwise ``~/.cache/scHopfield/base_grn``.
    """
    if cache_dir is not None:
        return os.path.expanduser(str(cache_dir))
    env = os.environ.get("SCHOPFIELD_CACHE")
    if env:
        return os.path.join(os.path.expanduser(env), "base_grn")
    return os.path.join(os.path.expanduser("~"), ".cache", "scHopfield", "base_grn")


def _url(spec: dict) -> str:
    return f"{_CELLORACLE_RAW}/{spec['commit']}/{spec['path']}"


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    """Hash a file without reading all of it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _manual_fallback(spec: dict, dest: str) -> str:
    """The two ways to get the file without this helper, named in every failure."""
    return (
        f"Two ways to get it by hand:\n"
        f"  1. Download it yourself and put it at this exact path:\n"
        f"       {dest}\n"
        f"       {_url(spec)}\n"
        f"  2. Install CellOracle and call celloracle.data.{spec['loader']}, then copy\n"
        f"     the file it leaves in ~/celloracle_data to the path above.\n"
        f"Note that CellOracle distributes this table for non-commercial academic use only; "
        f"see DATA_SOURCES.md."
    )


def _download(url: str, tmp_path: str) -> None:
    """Stream ``url`` into ``tmp_path``, raising :class:`OSError` on any transport error."""
    with urllib.request.urlopen(url, timeout=_TIMEOUT) as response:
        with open(tmp_path, "wb") as out:
            shutil.copyfileobj(response, out)


def _ensure_cached(name: str, cache_dir: Optional[str] = None) -> str:
    """Return a local path to a verified copy of a base GRN, downloading on a miss.

    The download lands in a temporary file and is checksummed before it is moved into
    place, so an interrupted or truncated transfer never becomes a cache entry. A file
    already in the cache is checksummed too, which costs about 30 ms and turns a
    corrupted or substituted cache into a loud failure rather than a wrong scaffold.
    """
    key = _resolve(name)
    spec = BASE_GRNS[key]
    root = _cache_root(cache_dir)
    dest = os.path.join(root, os.path.basename(spec["path"]))

    if os.path.isfile(dest):
        digest = _sha256(dest)
        if digest != spec["sha256"]:
            raise RuntimeError(
                f"the cached {key} base GRN does not match the checksum scHopfield pins.\n"
                f"  file:     {dest}\n"
                f"  expected: {spec['sha256']}\n"
                f"  found:    {digest}\n"
                f"This is not the table the published results were produced with, so it is "
                f"not used. Delete the file and call this function again to fetch a fresh "
                f"copy.\n{_manual_fallback(spec, dest)}"
            )
        return dest

    os.makedirs(root, exist_ok=True)
    handle, tmp_path = tempfile.mkstemp(dir=root, prefix=".download-", suffix=".part")
    os.close(handle)
    try:
        try:
            _download(_url(spec), tmp_path)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise RuntimeError(
                f"could not download the {key} base GRN ({spec['description']}).\n"
                f"  url:   {_url(spec)}\n"
                f"  cause: {type(exc).__name__}: {exc}\n"
                f"scHopfield does not ship this table, so there is no local copy to fall back "
                f"on.\n{_manual_fallback(spec, dest)}"
            ) from exc

        digest = _sha256(tmp_path)
        if digest != spec["sha256"]:
            n_got = os.path.getsize(tmp_path)
            raise RuntimeError(
                f"the downloaded {key} base GRN does not match the checksum scHopfield "
                f"pins, so it was discarded rather than cached.\n"
                f"  url:      {_url(spec)}\n"
                f"  expected: {spec['sha256']} ({spec['n_bytes']:,} bytes)\n"
                f"  found:    {digest} ({n_got:,} bytes)\n"
                f"Either the transfer was truncated, or the pinned content changed, which "
                f"should be impossible for a commit-pinned URL. Retry once; if it repeats, "
                f"the pin needs re-checking.\n{_manual_fallback(spec, dest)}"
            )
        os.replace(tmp_path, dest)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return dest


def fetch_base_grn(
    name: str = "mouse",
    cache_dir: Optional[str] = None,
    return_path: bool = False,
):
    """Download a CellOracle base GRN into a local cache and return it.

    scHopfield does not distribute a base gene regulatory network. This fetches one
    from the CellOracle repository at a pinned commit, verifies its sha256, caches it,
    and returns it in the wide format :func:`build_scaffold` consumes. The first call
    downloads; later calls read the cache, so a tutorial is offline-safe after its
    first run.

    **The fetched table is not covered by scHopfield's MIT license.** CellOracle
    distributes it under a modified Apache License 2.0 restricted to non-commercial
    academic purposes; for any other use, including commercial use, contact the Morris
    lab. Complying with those terms is the caller's responsibility. ``DATA_SOURCES.md``
    states the restriction in full and lists the works to cite.

    Parameters
    ----------
    name
        Which table to fetch: ``'mouse_atlas'``, ``'human_promoter'`` or
        ``'mouse_promoter'``. The species aliases ``'mouse'`` and ``'human'`` resolve
        to the table that species was fit with in the paper, so ``'mouse'`` is the
        atlas.
    cache_dir
        Directory to cache the download in. Defaults to ``$SCHOPFIELD_CACHE/base_grn``
        if that variable is set, else ``~/.cache/scHopfield/base_grn``.
    return_path
        Return the path to the cached file instead of the loaded table. Use this to
        fetch without a parquet engine, or to hand the file to another tool.

    Returns
    -------
    :class:`pandas.DataFrame` or str
        The wide base GRN: a ``gene_short_name`` column, a ``peak_id`` column, and one
        binary column per transcription factor. The cached path instead, if
        ``return_path=True``.

    Raises
    ------
    ValueError
        The name is not one of the three tables or two aliases.
    RuntimeError
        The download failed, or the bytes do not match the pinned checksum. Both
        messages name the two manual alternatives rather than surfacing a bare URL
        error, because there is no local copy to fall back on.

    Examples
    --------
    >>> import scHopfield as sch
    >>> base = sch.fetch_base_grn("mouse")                      # doctest: +SKIP
    >>> scaffold = sch.build_scaffold(adata, base)              # doctest: +SKIP

    See Also
    --------
    build_scaffold : Turn the returned table into a regulator-by-target scaffold.
    """
    path = _ensure_cached(name, cache_dir=cache_dir)
    if return_path:
        return path
    try:
        return pd.read_parquet(path)
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            f"reading {path} needs a parquet engine. Install one with "
            f"`pip install pyarrow`, or call this function with return_path=True and "
            f"read the file yourself."
        ) from exc
