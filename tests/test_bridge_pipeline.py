"""Integration tests for bridge_pipeline.py (Phase 2+3 orchestration)."""
import numpy as np
import faiss
import pytest

from arxanon import config
from arxanon.bridge_pipeline import run_bridge_pipeline
from arxanon.clusters import BridgePipelineResult, TDAResult
from arxanon.db import (
    init_db,
    store_embedding_idx,
    upsert_citation_edge,
    upsert_paper,
)


def _seed_bridge_db_and_faiss(n_cs: int = 6, n_math: int = 6) -> None:
    """Seed DB and FAISS with papers that produce ≥1 bridge cluster.

    Group cs: near basis vector e₀ = (1, 0, ...).
    Group math: near basis vector e₄ = (0, 0, 0, 0, 1, ...).

    Within-group cosine similarity ≈ 0.99 (>> SIMILARITY_THRESHOLD=0.72)
    → intra-group pairs form bridge edges (no citations seeded).

    Cross-group cosine similarity ≈ 0 (<< 0.72)
    → cross-group pairs never enter the similarity graph.

    HDBSCAN receives 12 bridge papers split into two tight, well-separated
    groups and reliably finds ≥1 cluster (min_cluster_size=3).
    """
    init_db()

    dim = 8
    rng = np.random.default_rng(42)
    entries: list[tuple[str, np.ndarray]] = []

    for i in range(n_cs):
        pid = f"cs{i + 1}"
        base = np.zeros(dim, dtype=np.float32)
        base[0] = 1.0                                    # cluster A: near e₀
        vec = base + rng.standard_normal(dim).astype(np.float32) * 0.04
        vec /= np.linalg.norm(vec)
        upsert_paper(pid, f"CS Paper {i + 1}", f"CS abstract {i + 1}.",
                     '["cs.LG"]', "2024-01-01", "semantic")
        entries.append((pid, vec))

    for i in range(n_math):
        pid = f"math{i + 1}"
        base = np.zeros(dim, dtype=np.float32)
        base[4] = 1.0                                    # cluster B: near e₄ (orthogonal)
        vec = base + rng.standard_normal(dim).astype(np.float32) * 0.04
        vec /= np.linalg.norm(vec)
        upsert_paper(pid, f"Math Paper {i + 1}", f"Math abstract {i + 1}.",
                     '["math.DS"]', "2024-01-01", "structural")
        entries.append((pid, vec))

    vecs = np.stack([v for _, v in entries])
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, str(config.FAISS_PATH))

    for i, (pid, _) in enumerate(entries):
        store_embedding_idx(pid, i)


# ── Empty-state tests ─────────────────────────────────────────────────────────

def test_no_crash_on_empty_db():
    """Empty DB (no papers, no FAISS file) → returns zeroed BridgePipelineResult."""
    init_db()
    result = run_bridge_pipeline(enable_tda=False)

    assert result.graph_nodes == 0
    assert result.graph_edges_full == 0
    assert result.graph_edges_bridge == 0
    assert result.clusters == []


def test_no_crash_on_no_embeddings():
    """Papers in DB but no FAISS file → returns zeroed BridgePipelineResult."""
    init_db()
    upsert_paper("p1", "Title", "Abstract", '["cs.LG"]', "2024-01-01", "test")
    # No embedding_index entry → idx_map is empty → early return

    result = run_bridge_pipeline(enable_tda=False)
    assert result.graph_nodes == 0


# ── Bibcoupling idempotency ───────────────────────────────────────────────────

def test_bibcoupling_idempotent():
    """Running the pipeline twice inserts 0 new bibcoupling edges on the second call."""
    init_db()
    for pid in ["pa", "pb", "ref1", "ref2", "ref3"]:
        upsert_paper(pid, f"Title {pid}", "Abstract", '["cs.LG"]', "2024-01-01", "test")

    upsert_citation_edge("pa", "ref1", "direct")
    upsert_citation_edge("pa", "ref2", "direct")
    upsert_citation_edge("pa", "ref3", "direct")
    upsert_citation_edge("pb", "ref1", "direct")
    upsert_citation_edge("pb", "ref2", "direct")
    upsert_citation_edge("pb", "ref3", "direct")

    result1 = run_bridge_pipeline(coupling_threshold=3, enable_tda=False)
    result2 = run_bridge_pipeline(coupling_threshold=3, enable_tda=False)

    assert result1.bibcoupling_edges_added == 2   # pa→pb and pb→pa
    assert result2.bibcoupling_edges_added == 0   # INSERT OR IGNORE: no new rows


# ── Full integration test ─────────────────────────────────────────────────────

def test_run_bridge_pipeline_with_seeded_db():
    """Seeded DB + FAISS → non-empty BridgePipelineResult with at least one cluster."""
    _seed_bridge_db_and_faiss(n_cs=6, n_math=6)

    result = run_bridge_pipeline(coupling_threshold=3, enable_tda=False)

    assert isinstance(result, BridgePipelineResult)
    assert result.graph_nodes == 12            # all papers appear in some sim edge
    # 15 intra-cs + 15 intra-math = 30 intra-group edges (cross-group below threshold)
    assert result.graph_edges_full == 30
    assert result.graph_edges_bridge == 30     # no citations seeded → none removed
    assert result.bibcoupling_edges_added == 0 # no direct edges were seeded
    assert not result.tda_result.enabled       # TDA disabled via flag
    assert len(result.clusters) >= 1

    for cluster in result.clusters:
        assert cluster.score.composite > 0.0
        assert len(cluster.paper_ids) >= config.HDBSCAN_MIN_CLUSTER_SIZE


def test_run_bridge_pipeline_stage_callbacks():
    """on_stage callback receives expected stage names."""
    _seed_bridge_db_and_faiss(n_cs=6, n_math=6)

    stages_seen: list[str] = []

    def _on_stage(stage: str, value) -> None:
        stages_seen.append(stage)

    run_bridge_pipeline(coupling_threshold=3, enable_tda=False, on_stage=_on_stage)

    assert "bibcoupling_done" in stages_seen
    assert "graph_done" in stages_seen
    assert "clustering_done" in stages_seen
