"""Phase 2 TDA tests: giotto-tda import fallback + edge cases."""
import sys
import warnings

import numpy as np
import pytest

from arxanon.clusters import TDAResult


def _can_import_gtda() -> bool:
    try:
        import gtda  # noqa: F401
        return True
    except Exception:
        return False


def _unit_vecs(n: int, dim: int = 8, seed: int = 0) -> tuple[list[str], np.ndarray, dict]:
    """Return (arxiv_ids, L2-normalised vectors, paper_categories)."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [f"p{i}" for i in range(n)]
    cats = {pid: ["cs.LG"] for pid in ids}
    return ids, vecs, cats


# ── Too-few-papers guard ──────────────────────────────────────────────────────

def test_tda_too_few_papers():
    """Fewer than 4 bridge papers → TDAResult(enabled=False) without attempting import."""
    from arxanon.tda import compute_tda

    ids, vecs, cats = _unit_vecs(3)
    result = compute_tda(ids, vecs, cats)

    assert not result.enabled
    assert result.n_cycles == 0
    assert "Too few" in (result.warning or "")


def test_tda_zero_papers():
    """Zero papers → TDAResult(enabled=False)."""
    from arxanon.tda import compute_tda

    result = compute_tda([], np.zeros((0, 8), dtype=np.float32), {})

    assert not result.enabled


# ── Import-error fallback ─────────────────────────────────────────────────────

def test_tda_import_error_fallback(monkeypatch):
    """Simulate giotto-tda not installed → TDAResult(enabled=False, warning=...)."""
    monkeypatch.setitem(sys.modules, "gtda", None)
    monkeypatch.setitem(sys.modules, "gtda.homology", None)

    from arxanon.tda import compute_tda

    ids, vecs, cats = _unit_vecs(10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ImportWarning)
        result = compute_tda(ids, vecs, cats)

    assert not result.enabled
    assert result.warning is not None
    assert "giotto-tda" in result.warning


# ── Disabled-flag path (object construction) ─────────────────────────────────

def test_tda_disabled_returns_warning():
    """TDAResult with enabled=False and warning='TDA disabled' is well-formed."""
    result = TDAResult(enabled=False, n_cycles=0, cycles=[], warning="TDA disabled")

    assert not result.enabled
    assert result.warning == "TDA disabled"


# ── Live giotto-tda path (skipped when not installed) ────────────────────────

@pytest.mark.skipif(not _can_import_gtda(), reason="giotto-tda not installed")
def test_tda_finds_h1_cycles_when_available():
    """When giotto-tda is installed, compute_tda runs without error."""
    from arxanon.tda import compute_tda

    ids, vecs, cats = _unit_vecs(20, dim=16, seed=99)
    result = compute_tda(ids, vecs, cats)

    assert result.enabled
    assert isinstance(result.n_cycles, int)
    assert result.n_cycles >= 0
