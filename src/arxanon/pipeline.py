"""Phase 1 pipeline: arXiv fetch → S2 citation graph → FAISS embeddings → similar pairs."""
from __future__ import annotations

import json
import logging
from typing import Callable, Optional

import faiss
import numpy as np

from . import config
from .arxiv_client import fetch_and_store_papers
from .db import (
    get_all_arxiv_ids,
    get_citation_edge_count,
    get_embedding_idx_map,
    get_paper,
    get_papers_without_embeddings,
    init_db,
    store_embedding_idx,
)
from .embedder import Embedder, load_or_create_faiss_index, save_faiss_index
from .semantic_scholar import fetch_and_store_citations

logger = logging.getLogger(__name__)

_STRUCTURAL_SUFFIX = "dynamical systems bifurcation stability analysis mathematical structure"


def default_structural_query(semantic_query: str) -> str:
    """Derive a generic structural companion query from a semantic query.

    Phase 3 will replace this with Gemma 4 structural decomposition.
    """
    return f"{semantic_query} {_STRUCTURAL_SUFFIX}"


def embed_and_index_papers() -> int:
    """Generate embeddings for any unindexed papers and update the FAISS index.

    Returns:
        Total number of vectors in the index after this call.
    """
    embedder = Embedder(config.EMBED_MODEL)
    papers = get_papers_without_embeddings()
    index = load_or_create_faiss_index(embedder.dim)

    if not papers:
        return index.ntotal

    abstracts = [p["abstract"] for p in papers]
    embeddings = embedder.encode(abstracts, is_query=False)

    start_idx = index.ntotal
    index.add(embeddings)

    for i, paper in enumerate(papers):
        store_embedding_idx(paper["arxiv_id"], start_idx + i)

    save_faiss_index(index)
    return index.ntotal


def find_top_similar_pairs(
    n: int = 5,
    cross_only: bool = True,
    tag_a: str = "semantic",
    tag_b: str = "structural",
) -> list[tuple]:
    """Find the top-n most similar paper pairs using FAISS batch search.

    Uses reconstruct_n + index.search for O(n·k) memory rather than O(n²).

    Args:
        n: Number of top pairs to return.
        cross_only: If True, only return pairs where one paper has tag_a and the other tag_b.
        tag_a: First query tag for cross-only filtering.
        tag_b: Second query tag for cross-only filtering.

    Returns:
        List of (id1, title1, primary_cat1, id2, title2, primary_cat2, score) tuples,
        sorted by score descending.
    """
    if not config.FAISS_PATH.exists():
        return []

    index = faiss.read_index(str(config.FAISS_PATH))
    total = index.ntotal
    if total < 2:
        return []

    vectors = np.zeros((total, index.d), dtype=np.float32)
    index.reconstruct_n(0, total, vectors)

    idx_map = get_embedding_idx_map()
    idx_to_arxiv = {v: k for k, v in idx_map.items()}

    idx_to_tag: dict[int, str] = {}
    for arxiv_id, faiss_idx in idx_map.items():
        p = get_paper(arxiv_id)
        if p:
            idx_to_tag[faiss_idx] = p.get("query_tag") or ""

    k = min(50, total)
    distances, neighbors = index.search(vectors, k)

    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int, float]] = []

    for i in range(total):
        for rank in range(k):
            j = int(neighbors[i, rank])
            if j == i or j < 0:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            if cross_only:
                ti = idx_to_tag.get(i, "")
                tj = idx_to_tag.get(j, "")
                if {ti, tj} != {tag_a, tag_b}:
                    continue
            seen.add(key)
            pairs.append((key[0], key[1], float(distances[i, rank])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    pairs = pairs[:n]

    result = []
    for idx1, idx2, score in pairs:
        id1 = idx_to_arxiv.get(idx1, "unknown")
        id2 = idx_to_arxiv.get(idx2, "unknown")
        p1 = get_paper(id1) or {}
        p2 = get_paper(id2) or {}
        result.append((
            id1,
            p1.get("title", ""),
            _primary_cat(p1.get("categories", "[]")),
            id2,
            p2.get("title", ""),
            _primary_cat(p2.get("categories", "[]")),
            score,
        ))

    return result


def load_papers_with_embeddings() -> dict[str, dict]:
    """Return all papers that have a FAISS embedding.

    Returns:
        Mapping of arxiv_id → paper dict for every paper in embedding_index.
    """
    idx_map = get_embedding_idx_map()
    result: dict[str, dict] = {}
    for arxiv_id in idx_map:
        paper = get_paper(arxiv_id)
        if paper:
            result[arxiv_id] = dict(paper)
    return result


def _primary_cat(categories_json: str) -> str:
    try:
        cats = json.loads(categories_json)
        return cats[0] if cats else "?"
    except (json.JSONDecodeError, IndexError):
        return "?"


def run_search(
    semantic_query: str,
    structural_query: str,
    max_results: int = 100,
    on_stage: Optional[Callable[[str, object], None]] = None,
) -> dict:
    """Run the full Phase 1 pipeline and return a results dict.

    Args:
        semantic_query: Query for papers in the researcher's field.
        structural_query: Query targeting papers from other domains with similar math structure.
        max_results: Maximum papers per query.
        on_stage: Optional callback(stage_name, value) for progress reporting.

    Returns:
        Dict with keys: semantic_count, structural_count, total_count, direct_edges,
        cocitation_edges, total_edges, index_size, embed_model, pairs.
    """

    def _report(stage: str, value: object) -> None:
        if on_stage:
            on_stage(stage, value)

    init_db()

    count_a = fetch_and_store_papers(
        semantic_query,
        max_results,
        "semantic",
        on_paper=lambda n: _report("semantic_paper", n),
    )
    _report("semantic_done", count_a)

    count_b = fetch_and_store_papers(
        structural_query,
        max_results,
        "structural",
        on_paper=lambda n: _report("structural_paper", n),
    )
    _report("structural_done", count_b)

    all_ids = get_all_arxiv_ids()
    fetch_and_store_citations(
        all_ids,
        on_paper=lambda done, total: _report("citation_paper", (done, total)),
    )
    _report("citations_done", None)

    index_size = embed_and_index_papers()
    _report("embeddings_done", index_size)

    pairs = find_top_similar_pairs(n=5, cross_only=True)

    edge_counts = get_citation_edge_count()

    return {
        "semantic_count": count_a,
        "structural_count": count_b,
        "total_count": count_a + count_b,
        "direct_edges": edge_counts.get("direct", 0),
        "cocitation_edges": edge_counts.get("cocitation", 0),
        "total_edges": sum(edge_counts.values()),
        "index_size": index_size,
        "embed_model": config.EMBED_MODEL,
        "pairs": pairs,
    }
