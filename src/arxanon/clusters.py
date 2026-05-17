"""Phase 2 dataclasses, bridge scoring, and HDBSCAN clustering.

All public dataclasses live here so every other module can import them
without circular dependencies.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import config

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def primary_cat(categories_json: str) -> str:
    """Extract the top-level arXiv category from a JSON-encoded categories list.

    '["cs.LG","stat.ML"]' → 'cs'
    '["math.DS"]'         → 'math'
    """
    try:
        cats = json.loads(categories_json)
        return cats[0].split(".")[0] if cats else "?"
    except (json.JSONDecodeError, IndexError, AttributeError):
        return "?"


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class BridgeScore:
    domain_diversity: float       # min(n_categories / 5, 1.0)
    structural_coherence: float   # mean pairwise cosine similarity within cluster
    citation_isolation: float     # fraction of paper pairs with no citation relationship
    topological_significance: float  # 0.0 or persistence of the best matching TDA cycle
    query_relevance: float        # mean cosine sim of cluster papers to query vector
    composite: float              # weighted sum per config weights


@dataclass
class PersistentCycle:
    cycle_id: int
    birth: float
    death: float
    persistence: float            # death - birth
    boundary_papers: list[str]    # approximate arxiv_ids near cycle boundary
    categories: list[str]         # distinct primary categories of boundary papers


@dataclass
class TDAResult:
    enabled: bool
    n_cycles: int
    cycles: list[PersistentCycle]
    warning: Optional[str] = None


@dataclass
class BridgePair:
    paper_a: str                  # arxiv_id
    paper_b: str                  # arxiv_id
    similarity: float             # cosine similarity score
    structure_a: dict             # from extract_formal_structure tool call
    structure_b: dict             # from extract_formal_structure tool call
    classification: str           # STRUCTURAL|METHODOLOGICAL|THEMATIC|SUPERFICIAL
    confidence_tier: str          # HIGH|MODERATE|EXPLORATORY
    reasoning_chain: str          # Gemma chain-of-thought (non-empty guaranteed)
    matched_properties: list[str]
    mismatched_properties: list[str]
    translation_hints: list[dict] = field(default_factory=list)  # [{term_a, term_b}]
    think_mode_used: bool = True  # True if think=True produced non-empty reasoning


@dataclass
class BridgeCluster:
    cluster_id: int
    paper_ids: list[str]
    categories: list[str]         # distinct primary categories
    score: BridgeScore
    bridge_edges: list[tuple]     # list of (id_a, id_b, cosine_sim)
    tda_cycle_ids: list[int]      # indices into TDAResult.cycles that overlap this cluster
    validated_pairs: list[BridgePair] = field(default_factory=list)
    confidence_tier: str = "EXPLORATORY"  # updated after Phase 3 validation


@dataclass
class BridgePipelineResult:
    graph_nodes: int
    graph_edges_full: int         # similarity graph edge count (before citation exclusion)
    graph_edges_bridge: int       # bridge edge count (after citation exclusion)
    bibcoupling_edges_added: int
    tda_result: TDAResult
    clusters: list[BridgeCluster] # sorted by composite score descending
    gemma_available: bool = False
    gemma_warning: Optional[str] = None
    query_relevance_scores: dict[str, float] = field(default_factory=dict)
    direct_cross_domain_pairs: list[BridgePair] = field(default_factory=list)
    structural_queries: list[str] = field(default_factory=list)
    semantic_queries: list[str] = field(default_factory=list)


# ── Scoring ───────────────────────────────────────────────────────────────────

def _confidence_tier(classification: str, n_matched: int) -> str:
    if classification == "STRUCTURAL" and n_matched >= 5:
        return "HIGH"
    if classification in ("STRUCTURAL", "METHODOLOGICAL") and n_matched >= 3:
        return "MODERATE"
    return "EXPLORATORY"


def _score_cluster(
    paper_ids: list[str],
    all_vectors: np.ndarray,
    idx_map: dict[str, int],
    papers: dict[str, dict],
    citation_pairs: set[frozenset],
    tda_result: TDAResult,
    query_vector: Optional[np.ndarray] = None,
) -> BridgeScore:
    n = len(paper_ids)

    # Domain diversity: distinct top-level arXiv categories, normalized by 5
    cats = {
        primary_cat(papers[pid]["categories"])
        for pid in paper_ids
        if pid in papers
    }
    domain_diversity = min(len(cats) / 5.0, 1.0)

    # Structural coherence: mean pairwise cosine similarity (L2-normalized → dot = cosine)
    valid_ids = [pid for pid in paper_ids if pid in idx_map]
    if len(valid_ids) > 1:
        vecs = np.array([all_vectors[idx_map[pid]] for pid in valid_ids])
        sims = vecs @ vecs.T
        n_v = len(valid_ids)
        structural_coherence = float((sims.sum() - n_v) / (n_v * (n_v - 1)))
        structural_coherence = max(0.0, min(1.0, structural_coherence))
    else:
        structural_coherence = 1.0

    # Citation isolation: fraction of paper pairs with no citation relationship
    pairs_list = [
        (a, b)
        for i, a in enumerate(paper_ids)
        for b in paper_ids[i + 1 :]
    ]
    if pairs_list:
        isolated = sum(
            1 for a, b in pairs_list if frozenset({a, b}) not in citation_pairs
        )
        citation_isolation = isolated / len(pairs_list)
    else:
        citation_isolation = 1.0

    # Topological significance: best matching cycle's persistence
    topo_sig = 0.0
    if tda_result.enabled and tda_result.cycles:
        pid_set = set(paper_ids)
        for cycle in tda_result.cycles:
            if len(pid_set & set(cycle.boundary_papers)) >= 2:
                topo_sig = min(1.0, cycle.persistence)
                break

    # Query relevance: mean cosine sim of cluster papers to query vector
    query_relevance = 0.0
    if query_vector is not None and valid_ids:
        q_sims = np.array([all_vectors[idx_map[pid]] for pid in valid_ids]) @ query_vector
        query_relevance = float(np.clip(np.mean(q_sims), 0.0, 1.0))

    composite = (
        config.BRIDGE_WEIGHT_DOMAIN * domain_diversity
        + config.BRIDGE_WEIGHT_COHERENCE * structural_coherence
        + config.BRIDGE_WEIGHT_ISOLATION * citation_isolation
        + config.BRIDGE_WEIGHT_TOPOLOGY * topo_sig
        + config.BRIDGE_WEIGHT_QUERY * query_relevance
    )
    return BridgeScore(
        domain_diversity=domain_diversity,
        structural_coherence=structural_coherence,
        citation_isolation=citation_isolation,
        topological_significance=topo_sig,
        query_relevance=query_relevance,
        composite=composite,
    )


# ── HDBSCAN clustering ────────────────────────────────────────────────────────

def run_hdbscan_and_score(
    bridge_paper_ids: list[str],
    all_vectors: np.ndarray,
    idx_map: dict[str, int],
    bridge_edges: list[tuple],
    papers: dict[str, dict],
    citation_pairs: set[frozenset],
    tda_result: TDAResult,
    query_vector: Optional[np.ndarray] = None,
) -> list[BridgeCluster]:
    """Cluster bridge-graph papers with HDBSCAN and score each cluster.

    Args:
        bridge_paper_ids: Papers appearing in at least one bridge edge.
        all_vectors: Full FAISS vector bank (ntotal, dim), L2-normalized float32.
        idx_map: Mapping of arxiv_id → faiss_idx.
        bridge_edges: List of (id_a, id_b, cosine_sim) tuples.
        papers: Mapping of arxiv_id → paper metadata dict.
        citation_pairs: Set of frozenset({id_a, id_b}) for all citation pairs.
        tda_result: Result from compute_tda() (may have enabled=False).

    Returns:
        List of BridgeCluster sorted by composite score descending.
        Empty list if too few papers or no clusters found.
    """
    import hdbscan as hdbscan_lib

    n = len(bridge_paper_ids)
    min_needed = config.HDBSCAN_MIN_CLUSTER_SIZE * 2
    if n < min_needed:
        logger.warning(
            "Too few bridge papers (%d) for HDBSCAN (need >= %d). "
            "Try --max-results with a larger value.",
            n, min_needed,
        )
        return []

    # Build sub-vector matrix for bridge papers
    valid_ids = [pid for pid in bridge_paper_ids if pid in idx_map]
    if len(valid_ids) < min_needed:
        return []

    vecs = np.array([all_vectors[idx_map[pid]] for pid in valid_ids])

    # Cosine distance matrix: 1.0 - cosine_sim (L2-normalized → dot = cosine)
    cos_sim = vecs @ vecs.T
    dist_matrix = 1.0 - cos_sim
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.clip(dist_matrix, 0.0, 2.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2  # ensure symmetry
    dist_matrix = dist_matrix.astype(np.float64)

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=config.HDBSCAN_MIN_SAMPLES,
        metric="precomputed",
    )
    labels = clusterer.fit_predict(dist_matrix)

    # Group papers by cluster label; label -1 = noise, excluded
    cluster_groups: dict[int, list[str]] = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        cluster_groups.setdefault(int(label), []).append(valid_ids[i])

    if not cluster_groups:
        logger.info("HDBSCAN produced no clusters (all papers labeled as noise).")
        return []

    bridge_edge_set: dict[frozenset, float] = {
        frozenset({a, b}): score for a, b, score in bridge_edges
    }

    clusters: list[BridgeCluster] = []
    for cluster_id, cids in cluster_groups.items():
        pid_set = set(cids)

        # Bridge edges that lie entirely within this cluster
        cluster_edges = [
            (a, b, s) for a, b, s in bridge_edges if a in pid_set and b in pid_set
        ]

        # Distinct primary categories
        categories = sorted({
            primary_cat(papers[pid]["categories"])
            for pid in cids
            if pid in papers
        })

        # TDA cycles overlapping this cluster
        tda_cycle_ids: list[int] = []
        if tda_result.enabled:
            for i, cycle in enumerate(tda_result.cycles):
                if len(pid_set & set(cycle.boundary_papers)) >= 2:
                    tda_cycle_ids.append(i)

        score = _score_cluster(
            paper_ids=cids,
            all_vectors=all_vectors,
            idx_map=idx_map,
            papers=papers,
            citation_pairs=citation_pairs,
            tda_result=tda_result,
            query_vector=query_vector,
        )

        clusters.append(BridgeCluster(
            cluster_id=cluster_id,
            paper_ids=cids,
            categories=categories,
            score=score,
            bridge_edges=cluster_edges,
            tda_cycle_ids=tda_cycle_ids,
        ))

    clusters.sort(key=lambda c: c.score.composite, reverse=True)
    return clusters
