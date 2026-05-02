"""Phase 2, Layer 1: full similarity graph + citation exclusion filter.

Builds a cosine-similarity graph over all embedded papers, then removes
every edge where a citation relationship (direct, co-citation, or bibcoupling)
exists between the two papers.  What remains is the candidate bridge graph.
"""
from __future__ import annotations

import numpy as np

from . import config


def build_bridge_graph(
    papers: dict[str, dict],
    idx_map: dict[str, int],
    all_vectors: np.ndarray,
    citation_pairs: set[frozenset],
    sim_threshold: float = config.SIMILARITY_THRESHOLD,
) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
    """Build the full similarity graph and return its bridge-filtered subset.

    For n <= 2000 papers uses a vectorised numpy matmul (exact pairwise).
    Vectors must already be L2-normalised (dot product == cosine similarity).

    Args:
        papers: Mapping of arxiv_id → paper metadata dict (must include all
                papers that appear in idx_map).
        idx_map: Mapping of arxiv_id → FAISS row index.
        all_vectors: Full FAISS vector bank, shape (ntotal, dim), float32.
        citation_pairs: Set of frozenset({id_a, id_b}) for every pair that has
                        ANY citation relationship.  Produced by
                        db.get_citation_pairs_for_nodes().
        sim_threshold: Minimum cosine similarity to include an edge.

    Returns:
        (similarity_edges, bridge_edges) where each element is a list of
        (id_a, id_b, cosine_similarity) triples sorted by similarity desc.

        similarity_edges: All pairs with sim >= sim_threshold.
        bridge_edges: similarity_edges minus any pair present in citation_pairs.
    """
    # Only include papers that have embeddings
    paper_ids = [pid for pid in papers if pid in idx_map]
    if len(paper_ids) < 2:
        return [], []

    vecs = np.array([all_vectors[idx_map[pid]] for pid in paper_ids], dtype=np.float32)

    # Pairwise cosine similarities (L2-normalised → dot product = cosine)
    cos_sim = vecs @ vecs.T  # (n, n)

    # Upper-triangle mask: pairs above threshold (excluding self-similarity on diagonal)
    mask = np.triu(cos_sim >= sim_threshold, k=1)
    i_indices, j_indices = np.where(mask)

    similarity_edges: list[tuple[str, str, float]] = [
        (paper_ids[i], paper_ids[j], float(cos_sim[i, j]))
        for i, j in zip(i_indices.tolist(), j_indices.tolist())
    ]
    similarity_edges.sort(key=lambda e: e[2], reverse=True)

    bridge_edges: list[tuple[str, str, float]] = [
        (a, b, s)
        for a, b, s in similarity_edges
        if frozenset({a, b}) not in citation_pairs
    ]

    return similarity_edges, bridge_edges
