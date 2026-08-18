"""Tests for the base GRN fetch helper: caching, checksums, and failure messages.

scHopfield does not distribute a base GRN, so ``fetch_base_grn`` is the only route to
one and there is no local copy to fall back on when it goes wrong. That makes two
behaviors load-bearing rather than incidental. A corrupted or substituted table must
never reach :func:`build_scaffold`, because it would produce a plausible scaffold from
the wrong prior and nothing downstream would notice. And every failure must name the
manual alternatives, since a bare ``URLError`` leaves a reader with no way forward.

Nothing here touches the network. The transport is stubbed and the registry is
monkeypatched with a small fake table, so the tests pin the cache and verification
logic rather than the availability of a third-party repository.
"""
import hashlib
import io
import os
import urllib.error

import pytest

from scHopfield.inference import base_grn


PAYLOAD = b"a base GRN would be here"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
FILENAME = "fake_base_GRN.parquet"


@pytest.fixture
def registered(monkeypatch):
    """Register a fake table so the tests never depend on a real download."""
    monkeypatch.setitem(
        base_grn.BASE_GRNS,
        "fake",
        dict(
            description="fake table for tests",
            commit="0" * 40,
            path=f"celloracle/data/{FILENAME}",
            sha256=DIGEST,
            n_bytes=len(PAYLOAD),
            loader="load_fake_base_GRN()",
        ),
    )
    return "fake"


def _serve(monkeypatch, payload):
    """Stub the transport so a download writes ``payload`` and touches no network."""
    def fake_download(url, tmp_path):
        with open(tmp_path, "wb") as out:
            out.write(payload)

    monkeypatch.setattr(base_grn, "_download", fake_download)


def _fail(monkeypatch, exc):
    """Stub the transport so any download attempt raises ``exc``."""
    def fake_download(url, tmp_path):
        raise exc

    monkeypatch.setattr(base_grn, "_download", fake_download)


# --------------------------------------------------------------------------- #
# Name resolution
# --------------------------------------------------------------------------- #
def test_species_aliases_resolve_to_the_paper_tables():
    # 'mouse' is ambiguous on its own: two mouse tables exist. It must resolve to the
    # atlas, which is the prior behind every mouse figure in the paper.
    assert base_grn._resolve("mouse") == "mouse_atlas"
    assert base_grn._resolve("human") == "human_promoter"
    assert base_grn._resolve("mouse_promoter") == "mouse_promoter"
    assert base_grn._resolve("MOUSE") == "mouse_atlas"


def test_unknown_name_lists_the_tables_and_the_aliases():
    with pytest.raises(ValueError) as err:
        base_grn._resolve("rat")
    message = str(err.value)
    for name in ("mouse_atlas", "human_promoter", "mouse_promoter"):
        assert name in message
    assert "mouse -> mouse_atlas" in message


def test_the_three_pinned_tables_are_fully_specified():
    # A missing checksum or an unpinned URL would silently disable verification.
    for key, spec in base_grn.BASE_GRNS.items():
        assert len(spec["sha256"]) == 64, key
        assert len(spec["commit"]) == 40, key
        assert spec["path"].endswith(".parquet"), key
        assert "/master/" not in base_grn._url(spec), key
        assert spec["commit"] in base_grn._url(spec), key


# --------------------------------------------------------------------------- #
# Cache behavior
# --------------------------------------------------------------------------- #
def test_download_verifies_then_caches(tmp_path, monkeypatch, registered):
    _serve(monkeypatch, PAYLOAD)
    path = base_grn._ensure_cached(registered, cache_dir=str(tmp_path))
    assert os.path.basename(path) == FILENAME
    assert open(path, "rb").read() == PAYLOAD


def test_a_cached_file_is_not_downloaded_again(tmp_path, monkeypatch, registered):
    _serve(monkeypatch, PAYLOAD)
    first = base_grn._ensure_cached(registered, cache_dir=str(tmp_path))
    # Any second download attempt now raises, so a cache hit is the only way through.
    _fail(monkeypatch, AssertionError("re-downloaded a file already in the cache"))
    assert base_grn._ensure_cached(registered, cache_dir=str(tmp_path)) == first


def test_cache_dir_falls_back_to_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHOPFIELD_CACHE", str(tmp_path))
    assert base_grn._cache_root(None) == os.path.join(str(tmp_path), "base_grn")
    monkeypatch.delenv("SCHOPFIELD_CACHE")
    assert base_grn._cache_root(None).startswith(os.path.expanduser("~"))
    # An explicit argument always wins over the environment.
    monkeypatch.setenv("SCHOPFIELD_CACHE", str(tmp_path))
    assert base_grn._cache_root("/somewhere/else") == "/somewhere/else"


# --------------------------------------------------------------------------- #
# Verification: the wrong table must never reach build_scaffold
# --------------------------------------------------------------------------- #
def test_a_truncated_download_is_rejected_and_not_cached(tmp_path, monkeypatch, registered):
    _serve(monkeypatch, PAYLOAD[:10])
    with pytest.raises(RuntimeError) as err:
        base_grn._ensure_cached(registered, cache_dir=str(tmp_path))
    assert "checksum" in str(err.value)
    # A bad transfer must leave nothing behind, neither the file nor its partial.
    assert os.listdir(tmp_path) == []


def test_a_corrupted_cache_names_the_file_to_delete(tmp_path, monkeypatch, registered):
    dest = tmp_path / FILENAME
    dest.write_bytes(b"not the table we pinned")
    _fail(monkeypatch, AssertionError("must not re-download over a corrupted cache"))
    with pytest.raises(RuntimeError) as err:
        base_grn._ensure_cached(registered, cache_dir=str(tmp_path))
    message = str(err.value)
    assert str(dest) in message
    assert "Delete the file" in message
    # The bad file is left in place, so the message's instruction is actionable and the
    # user can inspect what they actually have.
    assert dest.exists()


def test_a_substituted_cache_is_rejected_even_at_the_right_size(
    tmp_path, monkeypatch, registered
):
    # Size is not the check. A same-length file with different content is a different
    # prior, and it must not be handed to build_scaffold.
    (tmp_path / FILENAME).write_bytes(b"X" * len(PAYLOAD))
    _fail(monkeypatch, AssertionError("must not re-download"))
    with pytest.raises(RuntimeError):
        base_grn._ensure_cached(registered, cache_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Failure messages
# --------------------------------------------------------------------------- #
def test_network_failure_names_both_manual_fallbacks(tmp_path, monkeypatch, registered):
    _fail(monkeypatch, urllib.error.URLError("name resolution failed"))
    with pytest.raises(RuntimeError) as err:
        base_grn._ensure_cached(registered, cache_dir=str(tmp_path))
    message = str(err.value)
    # The URL to fetch by hand, the destination to put it at, the CellOracle call, and
    # the license restriction: a reader who cannot reach the network needs all four.
    assert "https://raw.githubusercontent.com/morris-lab/CellOracle/" in message
    assert str(tmp_path / FILENAME) in message
    assert "celloracle.data.load_fake_base_GRN()" in message
    assert "non-commercial" in message
    assert "name resolution failed" in message


def test_network_failure_chains_the_original_error(tmp_path, monkeypatch, registered):
    cause = urllib.error.URLError("connection refused")
    _fail(monkeypatch, cause)
    with pytest.raises(RuntimeError) as err:
        base_grn._ensure_cached(registered, cache_dir=str(tmp_path))
    assert err.value.__cause__ is cause


def test_an_http_error_is_reported_as_a_download_failure(tmp_path, monkeypatch, registered):
    # HTTPError is a URLError subclass, and a 404 is what a moved upstream file would
    # look like. It must produce the same guided message, not an unhandled traceback.
    _fail(
        monkeypatch,
        urllib.error.HTTPError("http://example.invalid", 404, "Not Found", {}, io.BytesIO()),
    )
    with pytest.raises(RuntimeError) as err:
        base_grn._ensure_cached(registered, cache_dir=str(tmp_path))
    assert "could not download" in str(err.value)


def test_a_timeout_is_reported_as_a_download_failure(tmp_path, monkeypatch, registered):
    _fail(monkeypatch, TimeoutError("timed out"))
    with pytest.raises(RuntimeError) as err:
        base_grn._ensure_cached(registered, cache_dir=str(tmp_path))
    assert "could not download" in str(err.value)


# --------------------------------------------------------------------------- #
# Public surface
# --------------------------------------------------------------------------- #
def test_return_path_skips_the_parquet_read(tmp_path, monkeypatch, registered):
    # The payload is not valid parquet, so this only passes if return_path avoids
    # parsing it. That is the escape hatch for an install with no parquet engine.
    _serve(monkeypatch, PAYLOAD)
    path = base_grn.fetch_base_grn(registered, cache_dir=str(tmp_path), return_path=True)
    assert os.path.isfile(path)


def test_fetch_base_grn_is_exported_at_the_top_level():
    import scHopfield as sch

    assert sch.fetch_base_grn is base_grn.fetch_base_grn
    assert "fetch_base_grn" in sch.__all__
    assert "fetch_base_grn" in sch.inf.__all__
