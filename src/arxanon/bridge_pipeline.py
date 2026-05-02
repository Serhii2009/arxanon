"""Phase 2 + 3 orchestration: bridge detection and Gemma 4 validation.

Public API:
    run_bridge_pipeline(...)  → BridgePipelineResult
    run_gemma_validation(...) → BridgePipelineResult (mutated in place)

Both functions are called by cli.py immediately after the Phase 1 pipeline.
They share the same BridgePipelineResult object so the CLI can display
everything after both phases complete.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

import faiss
import numpy as np

from . import config
from . import bridge_graph as bg_mod
from . import tda as tda_mod
from . import clusters as clusters_mod
from .clusters import BridgePipelineResult, TDAResult
from .db import (
    compute_and_store_bibcoupling,
    get_citation_pairs_for_nodes,
    get_embedding_idx_map,
)
from .pipeline import load_papers_with_embeddings

logger = logging.getLogger(__name__)


def run_bridge_pipeline(
    coupling_threshold: int = config.COUPLING_THRESHOLD,
    top_n_clusters: int = 5,
    enable_tda: bool = config.TDA_ENABLED,
    on_stage: Optional[Callable[[str, Any], None]] = None,
) -> BridgePipelineResult:
    """Run Phase 2: bibcoupling → bridge graph → TDA → HDBSCAN → scoring.

    Stage names emitted via on_stage(stage, value):
      'bibcoupling_done'  → int  (new edges added)
      'graph_done'        → (n_nodes, n_sim_edges, n_bridge_edges)
      'tda_done'          → TDAResult
      'clustering_done'   → int  (cluster count)

    Returns:
        BridgePipelineResult with gemma_available=False (set by run_gemma_validation).
    """
    def _emit(stage: str, value: Any) -> None:
        if on_stage:
            on_stage(stage, value)

    # ── Step 1: bibcoupling (idempotent) ──────────────────────────────────────
    bibcoupling_added = compute_and_store_bibcoupling(coupling_threshold)
    _emit("bibcoupling_done", bibcoupling_added)

    # ── Step 2: load papers + FAISS vectors ───────────────────────────────────
    papers = load_papers_with_embeddings()
    idx_map = get_embedding_idx_map()

    _empty = BridgePipelineResult(
        graph_nodes=0,
        graph_edges_full=0,
        graph_edges_bridge=0,
        bibcoupling_edges_added=bibcoupling_added,
        tda_result=TDAResult(enabled=False, n_cycles=0, cycles=[], warning="No embedded papers"),
        clusters=[],
    )

    if not idx_map:
        return _empty

    if not config.FAISS_PATH.exists():
        return _empty

    index = faiss.read_index(str(config.FAISS_PATH))
    if index.ntotal == 0:
        return _empty

    all_vectors = np.zeros((index.ntotal, index.d), dtype=np.float32)
    index.reconstruct_n(0, index.ntotal, all_vectors)

    # ── Step 3: citation pairs for citation exclusion filter ──────────────────
    all_paper_ids = list(idx_map.keys())
    citation_pairs = get_citation_pairs_for_nodes(all_paper_ids)

    # ── Step 4: build bridge graph ────────────────────────────────────────────
    sim_edges, bridge_edges = bg_mod.build_bridge_graph(
        papers=papers,
        idx_map=idx_map,
        all_vectors=all_vectors,
        citation_pairs=citation_pairs,
        sim_threshold=config.SIMILARITY_THRESHOLD,
    )
    n_nodes = len({pid for edge in sim_edges for pid in edge[:2]}) if sim_edges else len(papers)
    _emit("graph_done", (n_nodes, len(sim_edges), len(bridge_edges)))

    # ── Step 5: TDA (optional) ────────────────────────────────────────────────
    if enable_tda and bridge_edges:
        bridge_paper_ids = sorted({pid for edge in bridge_edges for pid in edge[:2]})
        bridge_vecs = np.array(
            [all_vectors[idx_map[pid]] for pid in bridge_paper_ids],
            dtype=np.float32,
        )
        paper_categories = {
            pid: json.loads(papers[pid]["categories"]) if pid in papers else []
            for pid in bridge_paper_ids
        }
        tda_result = tda_mod.compute_tda(
            arxiv_ids=bridge_paper_ids,
            vectors=bridge_vecs,
            paper_categories=paper_categories,
        )
    elif not enable_tda:
        tda_result = TDAResult(
            enabled=False, n_cycles=0, cycles=[], warning="TDA disabled"
        )
    else:
        tda_result = TDAResult(
            enabled=False, n_cycles=0, cycles=[], warning="No bridge edges for TDA"
        )
    _emit("tda_done", tda_result)

    # ── Step 6: HDBSCAN clustering + bridge scoring ───────────────────────────
    bridge_paper_ids = sorted({pid for edge in bridge_edges for pid in edge[:2]})
    cluster_list = clusters_mod.run_hdbscan_and_score(
        bridge_paper_ids=bridge_paper_ids,
        all_vectors=all_vectors,
        idx_map=idx_map,
        bridge_edges=bridge_edges,
        papers=papers,
        citation_pairs=citation_pairs,
        tda_result=tda_result,
    )
    _emit("clustering_done", len(cluster_list))

    return BridgePipelineResult(
        graph_nodes=n_nodes,
        graph_edges_full=len(sim_edges),
        graph_edges_bridge=len(bridge_edges),
        bibcoupling_edges_added=bibcoupling_added,
        tda_result=tda_result,
        clusters=cluster_list,
    )


def run_gemma_validation(
    result: BridgePipelineResult,
    papers: dict[str, dict],
    top_n_clusters: int = 5,
    on_pair: Optional[Callable[[int, int], None]] = None,
) -> BridgePipelineResult:
    """Run Phase 3: Gemma 4 structural analogy validation on top clusters.

    Mutates result.clusters[*].validated_pairs and result.clusters[*].confidence_tier.
    Sets result.gemma_available and result.gemma_warning.

    Returns:
        The same result object, updated in place.
    """
    from . import gemma_validator as gv

    available, error_msg = gv.check_ollama()
    if not available:
        result.gemma_available = False
        result.gemma_warning = (
            error_msg
            or f"Ollama not available.  Run: ollama serve && ollama pull {config.GEMMA_MODEL}"
        )
        return result

    result.gemma_available = True
    gv.validate_clusters(
        clusters=result.clusters,
        papers=papers,
        top_n_clusters=top_n_clusters,
        max_pairs_per_cluster=config.MAX_BRIDGE_VALIDATIONS,
        on_pair=on_pair,
    )
    return result
