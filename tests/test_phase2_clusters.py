"""Phase 2 cluster tests: primary_cat, bridge scoring, HDBSCAN, confidence tiers."""
import numpy as np
import pytest

from arxanon import config
from arxanon.clusters import (
    TDAResult,
    _confidence_tier,
    _score_cluster,
    primary_cat,
    run_hdbscan_and_score,
)


# ── primary_cat helper ────────────────────────────────────────────────────────

def test_primary_cat_cs():
    assert primary_cat('["cs.LG", "stat.ML"]') == "cs"


def test_primary_cat_math():
    assert primary_cat('["math.DS"]') == "math"


def test_primary_cat_nlin():
    assert primary_cat('["nlin.CD"]') == "nlin"


def test_primary_cat_empty_list():
    assert primary_cat("[]") == "?"


def test_primary_cat_malformed_json():
    assert primary_cat("not-json") == "?"


# ── Composite weight sanity ───────────────────────────────────────────────────

def test_composite_score_weights_sum_to_one():
    total = (
        config.BRIDGE_WEIGHT_DOMAIN
        + config.BRIDGE_WEIGHT_COHERENCE
        + config.BRIDGE_WEIGHT_ISOLATION
        + config.BRIDGE_WEIGHT_TOPOLOGY
        + config.BRIDGE_WEIGHT_QUERY
    )
    assert abs(total - 1.0) < 1e-9


# ── Confidence tier logic ─────────────────────────────────────────────────────

def test_confidence_tier_structural_high():
    assert _confidence_tier("STRUCTURAL", 5) == "HIGH"


def test_confidence_tier_structural_moderate():
    assert _confidence_tier("STRUCTURAL", 3) == "MODERATE"


def test_confidence_tier_methodological_moderate():
    assert _confidence_tier("METHODOLOGICAL", 3) == "MODERATE"


def test_confidence_tier_methodological_exploratory():
    assert _confidence_tier("METHODOLOGICAL", 2) == "EXPLORATORY"


def test_confidence_tier_thematic_exploratory():
    assert _confidence_tier("THEMATIC", 10) == "EXPLORATORY"


def test_confidence_tier_structural_not_enough_matched():
    assert _confidence_tier("STRUCTURAL", 2) == "EXPLORATORY"


# ── _score_cluster ────────────────────────────────────────────────────────────

def _eye_setup(n: int, categories: list[str]) -> tuple:
    """n orthogonal unit vectors, one per paper."""
    dim = max(n, 4)
    all_vectors = np.eye(dim, dtype=np.float32)
    paper_ids = [f"p{i}" for i in range(n)]
    idx_map = {pid: i for i, pid in enumerate(paper_ids)}
    papers = {
        pid: {"categories": f'["{cat}"]'}
        for pid, cat in zip(paper_ids, categories)
    }
    return paper_ids, all_vectors, idx_map, papers


def test_bridge_score_domain_diversity_max():
    """5 distinct top-level categories → domain_diversity == 1.0."""
    paper_ids, vecs, idx_map, papers = _eye_setup(
        5, ["cs.LG", "math.DS", "stat.ML", "nlin.CD", "physics.flu-dyn"]
    )
    score = _score_cluster(
        paper_ids=paper_ids,
        all_vectors=vecs,
        idx_map=idx_map,
        papers=papers,
        citation_pairs=set(),
        tda_result=TDAResult(enabled=False, n_cycles=0, cycles=[]),
    )
    assert score.domain_diversity == 1.0


def test_bridge_score_domain_diversity_partial():
    """2 distinct categories out of 5 → domain_diversity == 0.4."""
    paper_ids, vecs, idx_map, papers = _eye_setup(
        4, ["cs.LG", "cs.AI", "cs.RO", "cs.CV"]
    )
    score = _score_cluster(
        paper_ids=paper_ids,
        all_vectors=vecs,
        idx_map=idx_map,
        papers=papers,
        citation_pairs=set(),
        tda_result=TDAResult(enabled=False, n_cycles=0, cycles=[]),
    )
    # All 4 papers are "cs" → 1 distinct category → 1/5 = 0.2
    assert abs(score.domain_diversity - 0.2) < 1e-6


def test_bridge_score_citation_isolation_full():
    """No citation pairs → citation_isolation == 1.0."""
    paper_ids, vecs, idx_map, papers = _eye_setup(3, ["cs.LG", "cs.LG", "cs.LG"])
    score = _score_cluster(
        paper_ids=paper_ids,
        all_vectors=vecs,
        idx_map=idx_map,
        papers=papers,
        citation_pairs=set(),
        tda_result=TDAResult(enabled=False, n_cycles=0, cycles=[]),
    )
    assert score.citation_isolation == 1.0


def test_bridge_score_citation_isolation_partial():
    """1 of 3 pairs cited → citation_isolation == 2/3."""
    paper_ids, vecs, idx_map, papers = _eye_setup(3, ["cs.LG"] * 3)
    citation_pairs = {frozenset({"p0", "p1"})}  # one pair cited
    score = _score_cluster(
        paper_ids=paper_ids,
        all_vectors=vecs,
        idx_map=idx_map,
        papers=papers,
        citation_pairs=citation_pairs,
        tda_result=TDAResult(enabled=False, n_cycles=0, cycles=[]),
    )
    assert abs(score.citation_isolation - 2 / 3) < 1e-6


def test_bridge_score_composite_in_range():
    """Composite score is in [0, 1]."""
    paper_ids, vecs, idx_map, papers = _eye_setup(
        4, ["cs.LG", "math.DS", "stat.ML", "nlin.CD"]
    )
    score = _score_cluster(
        paper_ids=paper_ids,
        all_vectors=vecs,
        idx_map=idx_map,
        papers=papers,
        citation_pairs=set(),
        tda_result=TDAResult(enabled=False, n_cycles=0, cycles=[]),
    )
    assert 0.0 <= score.composite <= 1.0


# ── run_hdbscan_and_score ─────────────────────────────────────────────────────

def _make_two_group_data():
    """
    6 papers in 2 well-separated groups (3 each).
    Group A near e₀ = (1,0,...), Group B near e₄ = (0,0,0,0,1,...).
    Within-group cosine distance ≈ 0.005, across-group ≈ 1.0.
    """
    dim = 8
    raw = np.array([
        # Group A — cluster 0
        [1.0, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [1.0, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00],
        [1.0, 0.10, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00],
        # Group B — cluster 1
        [0.0, 0.00, 0.00, 0.00, 1.00, 0.10, 0.00, 0.00],
        [0.0, 0.00, 0.00, 0.00, 1.00, 0.00, 0.10, 0.00],
        [0.0, 0.00, 0.00, 0.00, 1.00, 0.10, 0.10, 0.00],
    ], dtype=np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)

    ids = ["a0", "a1", "a2", "b0", "b1", "b2"]
    idx_map = {pid: i for i, pid in enumerate(ids)}
    papers = {
        "a0": {"categories": '["cs.LG"]'},
        "a1": {"categories": '["math.DS"]'},
        "a2": {"categories": '["nlin.CD"]'},
        "b0": {"categories": '["stat.ML"]'},
        "b1": {"categories": '["physics.flu-dyn"]'},
        "b2": {"categories": '["econ.EM"]'},
    }
    # Bridge edges within each group (similarity after citation exclusion)
    bridge_edges = [
        ("a0", "a1", 0.995), ("a0", "a2", 0.990), ("a1", "a2", 0.985),
        ("b0", "b1", 0.995), ("b0", "b2", 0.990), ("b1", "b2", 0.985),
    ]
    tda_result = TDAResult(enabled=False, n_cycles=0, cycles=[])
    return ids, raw, idx_map, papers, bridge_edges, tda_result


def test_hdbscan_two_clear_clusters():
    """6 papers in 2 well-separated groups → 2 clusters with expected paper sets."""
    ids, vecs, idx_map, papers, bridge_edges, tda_result = _make_two_group_data()

    clusters = run_hdbscan_and_score(
        bridge_paper_ids=ids,
        all_vectors=vecs,
        idx_map=idx_map,
        bridge_edges=bridge_edges,
        papers=papers,
        citation_pairs=set(),
        tda_result=tda_result,
    )

    assert len(clusters) == 2

    found_sets = {frozenset(c.paper_ids) for c in clusters}
    assert frozenset({"a0", "a1", "a2"}) in found_sets
    assert frozenset({"b0", "b1", "b2"}) in found_sets


def test_hdbscan_sorted_by_composite_desc():
    """Clusters are returned sorted by composite score descending."""
    ids, vecs, idx_map, papers, bridge_edges, tda_result = _make_two_group_data()

    clusters = run_hdbscan_and_score(
        bridge_paper_ids=ids,
        all_vectors=vecs,
        idx_map=idx_map,
        bridge_edges=bridge_edges,
        papers=papers,
        citation_pairs=set(),
        tda_result=tda_result,
    )

    scores = [c.score.composite for c in clusters]
    assert scores == sorted(scores, reverse=True)


def test_hdbscan_too_few_papers():
    """Fewer papers than min_needed → empty cluster list."""
    dim = 4
    ids = ["p0", "p1"]  # only 2, need at least HDBSCAN_MIN_CLUSTER_SIZE * 2 = 6
    vecs = np.eye(dim, dtype=np.float32)[:2]
    idx_map = {"p0": 0, "p1": 1}
    papers = {pid: {"categories": '["cs.LG"]'} for pid in ids}

    clusters = run_hdbscan_and_score(
        bridge_paper_ids=ids,
        all_vectors=vecs,
        idx_map=idx_map,
        bridge_edges=[("p0", "p1", 0.9)],
        papers=papers,
        citation_pairs=set(),
        tda_result=TDAResult(enabled=False, n_cycles=0, cycles=[]),
    )

    assert clusters == []
