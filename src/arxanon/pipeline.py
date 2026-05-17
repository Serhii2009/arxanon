"""Phase 1 pipeline: arXiv fetch → S2 citation graph → FAISS embeddings → similar pairs."""
from __future__ import annotations

import json
import logging
import re
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

def _gemma_expand_queries(query: str) -> list[str]:
    """Expand a research question into 3 targeted arXiv search queries via LLM.

    Falls back to [query] (single query) if the LLM is unavailable or response is unparseable.
    """
    logger.debug("_gemma_expand_queries called with: %r", query)
    try:
        from .llm_client import call_llm
        logger.debug("Waiting for LLM query expansion (up to 45s)...")
        prompt = (
            "You are helping search arXiv for papers about a research problem. "
            "Generate 3 specific arXiv search queries that will find the most relevant papers.\n\n"
            "Rules:\n"
            "- All 3 queries must be about the SAME topic as the research problem, just from different angles\n"
            "- Use exact technical terms that appear in paper titles and abstracts\n"
            "- Each query is 3-6 words maximum\n"
            "- Do NOT change the subject domain\n"
            "- Focus on TRAINING DYNAMICS — papers studying temporal evolution DURING training, "
            "not static behavior of trained models\n"
            "- Good terms: 'during training', 'training dynamics', 'learning trajectory', "
            "'gradient dynamics', 'optimization trajectory', 'emergence during training'\n"
            "- Avoid: applications, inference-time behavior, interpretability of trained models\n"
            "- Generate queries that will find the most specific ML papers studying "
            "this phenomenon during training or inference. Use technical ML vocabulary. "
            "Target papers about training dynamics, emergence, or architectural behavior "
            "— not general topic papers.\n\n"
            "Examples for \"why are LLMs black boxes\":\n"
            "  mechanistic interpretability transformer circuits\n"
            "  feature visualization language model neurons\n"
            "  probing classifiers internal representations LLM\n\n"
            "Examples for \"protein folding thermodynamics\":\n"
            "  protein folding free energy landscape\n"
            "  thermodynamic stability protein structure\n"
            "  folding funnel kinetics molecular dynamics\n\n"
            f"Research problem: {query}\n\n"
            "Reply with exactly 3 queries, one per line.\n"
            "No numbering. No labels. No explanation."
        )
        raw = call_llm(prompt, timeout=45, temperature=0.2)
        logger.debug("LLM raw response:\n---\n%s\n---", raw)
        lines = [re.sub(r"^\d+[\.\)]\s*", "", ln).strip() for ln in raw.splitlines()]
        queries = [ln for ln in lines if len(ln.split()) >= 2]
        logger.debug("Parsed %d queries: %s", len(queries), queries)
        if 2 <= len(queries) <= 5:
            logger.debug("Using LLM-expanded queries: %s", queries[:4])
            return queries[:4]
        logger.debug("Query count %d outside [2,5] — falling back", len(queries))
    except Exception as exc:
        logger.debug("LLM unavailable or failed: %r — using fallback", exc)
    logger.debug("Fallback: single query [%r]", query)
    return [query]


def _llm_structural_queries(query: str, semantic_queries: list[str]) -> list[str]:
    """Generate queries targeting non-cs domains that share the same mathematical structure.

    Given the researcher's problem and the semantic queries already generated, asks the LLM
    to identify the abstract mathematical structure (equations, dynamical properties,
    topological features) and produce queries using vocabulary from math.DS, nlin.CD,
    physics.*, q-bio.* — never cs.* vocabulary.

    Returns up to 3 queries, or [] if the LLM fails.
    """
    logger.debug("_llm_structural_queries called for: %r", query)
    try:
        from .llm_client import call_llm
        logger.debug("Waiting for LLM structural query generation (up to 45s)...")
        sem_text = "\n".join(f"- {q}" for q in semantic_queries)
        prompt = (
            "A machine learning researcher has described the following phenomenon:\n\n"
            f"{query}\n\n"
            "Your task is to generate 3 arXiv search queries that will find papers "
            "from fields OUTSIDE machine learning that formally study this phenomenon.\n\n"
            "Step 1 — Identify which non-ML scientific field(s) most naturally "
            "contain formal mathematical or empirical studies of this phenomenon.\n\n"
            "Consider ALL of these fields and choose the most appropriate:\n"
            "- Neuroscience and cognitive neuroscience (q-bio.NC)\n"
            "- Cognitive psychology and learning theory (q-bio.NC, cs.HC)\n"
            "- Information theory and statistical inference (cs.IT, math.IT)\n"
            "- Control theory and systems engineering (math.OC, eess.SY)\n"
            "- Statistical physics and complex systems (cond-mat.stat-mech)\n"
            "- Nonlinear dynamics and chaos (nlin, math.DS)\n"
            "- Evolutionary biology and population dynamics (q-bio.PE)\n"
            "- Economics and game theory (econ, q-fin)\n"
            "- Linguistics and formal language theory (non-ML cs.CL papers)\n"
            "- Numerical analysis and scientific computing (math.NA)\n"
            "- Probability theory and stochastic processes (math.PR)\n"
            "- Epistemology and formal logic (math.LO)\n"
            "- Sociology and network science (physics.soc-ph)\n\n"
            "Step 2 — Generate exactly 3 search queries using the native "
            "vocabulary of the field(s) you identified.\n\n"
            "Rules:\n"
            "- Each query must be 4-8 words\n"
            "- Each query must use vocabulary that a researcher in that field "
            "would use, NOT machine learning vocabulary\n"
            "- Queries must target paper titles and abstracts from that field\n"
            "- Do NOT default to dynamical systems unless the phenomenon "
            "genuinely involves nonlinear dynamics or bifurcations\n"
            "- Do NOT use quantum computing vocabulary\n"
            "- The 3 queries may target 1 field or 3 different fields, "
            "depending on what is most relevant to the phenomenon\n\n"
            "Output ONLY the 3 queries, one per line, between --- markers:\n"
            "---\n"
            "[query 1]\n"
            "[query 2]\n"
            "[query 3]\n"
            "---"
        )
        raw = call_llm(prompt, timeout=45, temperature=0.2)
        logger.debug("Structural LLM raw response:\n---\n%s\n---", raw)
        lines_raw = raw.splitlines()
        try:
            start = next(i for i, l in enumerate(lines_raw) if l.strip() == "---")
            end   = next(i for i, l in enumerate(lines_raw) if i > start and l.strip() == "---")
            lines_raw = lines_raw[start + 1 : end]
        except StopIteration:
            pass
        lines = [re.sub(r"^\d+[\.\)]\s*", "", ln).strip() for ln in lines_raw]
        queries = [ln for ln in lines if len(ln.split()) >= 2]
        logger.debug("Parsed %d structural queries: %s", len(queries), queries)
        if 2 <= len(queries) <= 5:
            logger.debug("Using structural queries: %s", queries[:3])
            return queries[:3]
        logger.debug("Structural query count %d outside [2,5] — returning empty", len(queries))
    except Exception as exc:
        logger.debug("Structural LLM failed: %r", exc)
    return []


def embed_and_index_papers(query: str = "") -> tuple[int, Optional[np.ndarray]]:
    """Generate embeddings for any unindexed papers and update the FAISS index.

    If query is provided and papers are embedded in this call, the query is also
    encoded while the model is still resident — avoiding a second model load later.

    Returns:
        (total vectors in index, query_vector or None)
    """
    embedder = Embedder(config.EMBED_MODEL)
    papers = get_papers_without_embeddings()
    index = load_or_create_faiss_index(embedder.dim)

    if not papers:
        return index.ntotal, None

    abstracts = [p["abstract"] for p in papers]
    embeddings = embedder.encode(abstracts, is_query=False)

    start_idx = index.ntotal
    index.add(embeddings)

    for i, paper in enumerate(papers):
        store_embedding_idx(paper["arxiv_id"], start_idx + i)

    save_faiss_index(index)

    query_vector: Optional[np.ndarray] = None
    if query:
        try:
            q_vecs = embedder.encode([query], is_query=True)
            q_raw = np.array(q_vecs[0], dtype=np.float32)
            norm = np.linalg.norm(q_raw)
            query_vector = q_raw / norm if norm > 0 else q_raw
        except Exception as exc:
            logger.warning("Could not embed query: %s", exc)
    return index.ntotal, query_vector


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

    index_size, _ = embed_and_index_papers()
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
