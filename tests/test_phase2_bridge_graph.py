"""Phase 2 bridge graph tests: bibcoupling + citation exclusion filter."""
import json

import numpy as np
import pytest

from arxanon import config
from arxanon.bridge_graph import build_bridge_graph
from arxanon.db import (
    compute_and_store_bibcoupling,
    get_citation_pairs_for_nodes,
    init_db,
    upsert_citation_edge,
    upsert_paper,
)


def _l2(v: list[float]) -> np.ndarray:
    arr = np.array(v, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def _make_papers(*arxiv_ids: str, cat: str = "cs.LG") -> dict:
    return {pid: {"categories": f'["{cat}"]', "query_tag": "test"} for pid in arxiv_ids}


# ── Bibcoupling threshold tests ───────────────────────────────────────────────

def test_bibcoupling_below_threshold():
    """Papers sharing 2 references do not get a bibcoupling edge at threshold=3."""
    init_db()
    for pid in ["pa", "pb", "ref1", "ref2"]:
        upsert_paper(pid, f"Title {pid}", "Abstract", '["cs.LG"]', "2024-01-01", "test")
    upsert_citation_edge("pa", "ref1", "direct")
    upsert_citation_edge("pa", "ref2", "direct")
    upsert_citation_edge("pb", "ref1", "direct")
    upsert_citation_edge("pb", "ref2", "direct")

    added = compute_and_store_bibcoupling(coupling_threshold=3)
    assert added == 0


def test_bibcoupling_at_threshold():
    """Papers sharing exactly 3 references get a bibcoupling edge (both directions)."""
    init_db()
    for pid in ["pa", "pb", "ref1", "ref2", "ref3"]:
        upsert_paper(pid, f"Title {pid}", "Abstract", '["cs.LG"]', "2024-01-01", "test")
    upsert_citation_edge("pa", "ref1", "direct")
    upsert_citation_edge("pa", "ref2", "direct")
    upsert_citation_edge("pa", "ref3", "direct")
    upsert_citation_edge("pb", "ref1", "direct")
    upsert_citation_edge("pb", "ref2", "direct")
    upsert_citation_edge("pb", "ref3", "direct")

    added = compute_and_store_bibcoupling(coupling_threshold=3)
    assert added == 2  # pa→pb and pb→pa


def test_bibcoupling_idempotent_standalone():
    """Calling compute_and_store_bibcoupling twice adds 0 edges on the second call."""
    init_db()
    for pid in ["pa", "pb", "r1", "r2", "r3"]:
        upsert_paper(pid, f"Title {pid}", "Abstract", '["cs.LG"]', "2024-01-01", "test")
    for ref in ["r1", "r2", "r3"]:
        upsert_citation_edge("pa", ref, "direct")
        upsert_citation_edge("pb", ref, "direct")

    first = compute_and_store_bibcoupling(coupling_threshold=3)
    second = compute_and_store_bibcoupling(coupling_threshold=3)
    assert first == 2
    assert second == 0


# ── Citation exclusion filter tests ──────────────────────────────────────────

def _simple_graph(extra_edges: list[tuple[str, str, str]] | None = None):
    """Three papers: pa ≈ pb (high sim), pc dissimilar. Returns (papers, idx_map, vecs)."""
    init_db()
    ids = ["pa", "pb", "pc"]
    vecs = np.array([
        _l2([1.0, 0.0, 0.0, 0.0]),
        _l2([0.999, 0.045, 0.0, 0.0]),
        _l2([0.0, 0.0, 0.0, 1.0]),
    ])
    for pid in ids:
        upsert_paper(pid, f"Title {pid}", "Abstract", '["cs.LG"]', "2024-01-01", "test")
    if extra_edges:
        for citing, cited, etype in extra_edges:
            upsert_citation_edge(citing, cited, etype)
    papers = _make_papers(*ids)
    idx_map = {pid: i for i, pid in enumerate(ids)}
    return papers, idx_map, vecs, ids


def test_citation_exclusion_removes_direct():
    """A pair connected by a direct citation is removed from bridge_edges."""
    papers, idx_map, vecs, ids = _simple_graph([("pa", "pb", "direct")])
    citation_pairs = get_citation_pairs_for_nodes(ids)

    _, bridge_edges = build_bridge_graph(
        papers=papers,
        idx_map=idx_map,
        all_vectors=vecs,
        citation_pairs=citation_pairs,
        sim_threshold=0.5,
    )
    bridge_pairs = {frozenset({a, b}) for a, b, _ in bridge_edges}
    assert frozenset({"pa", "pb"}) not in bridge_pairs


def test_citation_exclusion_removes_cocitation():
    """A pair connected by a cocitation edge is removed from bridge_edges."""
    papers, idx_map, vecs, ids = _simple_graph([("pa", "pb", "cocitation")])
    citation_pairs = get_citation_pairs_for_nodes(ids)

    _, bridge_edges = build_bridge_graph(
        papers=papers,
        idx_map=idx_map,
        all_vectors=vecs,
        citation_pairs=citation_pairs,
        sim_threshold=0.5,
    )
    bridge_pairs = {frozenset({a, b}) for a, b, _ in bridge_edges}
    assert frozenset({"pa", "pb"}) not in bridge_pairs


def test_citation_exclusion_preserves_uncited():
    """A high-similarity pair with no citation relationship survives in bridge_edges."""
    papers, idx_map, vecs, ids = _simple_graph()  # no citations
    citation_pairs = get_citation_pairs_for_nodes(ids)

    sim_edges, bridge_edges = build_bridge_graph(
        papers=papers,
        idx_map=idx_map,
        all_vectors=vecs,
        citation_pairs=citation_pairs,
        sim_threshold=0.5,
    )
    bridge_pairs = {frozenset({a, b}) for a, b, _ in bridge_edges}
    assert frozenset({"pa", "pb"}) in bridge_pairs


def test_bridge_edges_subset_of_similarity_edges():
    """bridge_edges ⊆ similarity_edges always holds."""
    init_db()
    n = 6
    rng = np.random.default_rng(7)
    ids = [f"p{i}" for i in range(n)]
    vecs = rng.standard_normal((n, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    for pid in ids:
        upsert_paper(pid, f"T {pid}", "A", '["cs.LG"]', "2024-01-01", "t")
    upsert_citation_edge("p0", "p1", "direct")
    upsert_citation_edge("p2", "p3", "cocitation")

    papers = _make_papers(*ids)
    idx_map = {pid: i for i, pid in enumerate(ids)}
    citation_pairs = get_citation_pairs_for_nodes(ids)

    sim_edges, bridge_edges = build_bridge_graph(
        papers=papers,
        idx_map=idx_map,
        all_vectors=vecs,
        citation_pairs=citation_pairs,
        sim_threshold=0.0,
    )

    sim_set = {frozenset({a, b}) for a, b, _ in sim_edges}
    for a, b, _ in bridge_edges:
        assert frozenset({a, b}) in sim_set


def test_build_bridge_graph_too_few_papers():
    """Fewer than 2 papers → both lists are empty."""
    init_db()
    ids = ["solo"]
    vec = _l2([1.0, 0.0, 0.0, 0.0])
    vecs = vec.reshape(1, -1)
    upsert_paper("solo", "Solo", "Abstract", '["cs.LG"]', "2024-01-01", "test")

    papers = _make_papers("solo")
    idx_map = {"solo": 0}
    citation_pairs = get_citation_pairs_for_nodes(ids)

    sim_edges, bridge_edges = build_bridge_graph(
        papers=papers,
        idx_map=idx_map,
        all_vectors=vecs,
        citation_pairs=citation_pairs,
        sim_threshold=0.0,
    )
    assert sim_edges == []
    assert bridge_edges == []
