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
import re
from typing import Any, Callable, Optional

import faiss
import numpy as np

from . import config
from . import bridge_graph as bg_mod
from . import tda as tda_mod
from . import clusters as clusters_mod
from .clusters import BridgePair, BridgePipelineResult, TDAResult, _confidence_tier, primary_cat
from .db import (
    compute_and_store_bibcoupling,
    get_citation_pairs_for_nodes,
    get_embedding_idx_map,
)
from .pipeline import load_papers_with_embeddings

logger = logging.getLogger(__name__)


def _primary_cats_from_papers(papers: dict) -> set[str]:
    """Return the set of distinct top-level arXiv categories across all papers."""
    cats: set[str] = set()
    for p in papers.values():
        try:
            cat_list = json.loads(p.get("categories", "[]") or "[]")
            if cat_list:
                cats.add(cat_list[0].split(".")[0])
        except (json.JSONDecodeError, IndexError):
            pass
    return cats


def _try_domain_expansion(
    query: str,
    papers: dict,
    _emit: Callable[[str, Any], None],
) -> bool:
    """Ask LLM for adjacent scientific domains, fetch papers, and embed them.

    Returns True if papers from a new top-level category were successfully added.
    """
    if not query:
        return False
    try:
        from .llm_client import call_llm
        from .arxiv_client import fetch_and_store_papers
        from .pipeline import embed_and_index_papers

        prompt = (
            f'A researcher is working on: "{query}"\n\n'
            "All papers retrieved so far are from the same scientific domain. "
            "Identify exactly 2 arXiv search queries targeting papers from DIFFERENT scientific "
            "fields (not computer science) that use the same mathematical structures.\n\n"
            "Examples of adjacent fields: statistical mechanics, control theory, "
            "signal processing, dynamical systems, information theory, computational biology.\n\n"
            "Reply with exactly 2 search queries, one per line. "
            "Each query is 3-6 words. No numbering. No labels. No explanation."
        )
        raw = call_llm(prompt, timeout=30, temperature=0.2)
        lines = [re.sub(r"^\d+[\.\)]\s*", "", ln).strip() for ln in raw.splitlines()]
        adj_queries = [ln for ln in lines if len(ln.split()) >= 2][:2]

        if not adj_queries:
            return False

        _emit("domain_expansion", adj_queries)
        n_per = max(20, len(papers) // 3)

        for i, q in enumerate(adj_queries):
            fetch_and_store_papers(q, n_per, f"adj{i + 1}")

        embed_and_index_papers()
        return True

    except Exception as exc:
        logger.warning("Domain expansion failed: %s", exc)
        return False


def run_bridge_pipeline(
    coupling_threshold: int = config.COUPLING_THRESHOLD,
    enable_tda: bool = config.TDA_ENABLED,
    on_stage: Optional[Callable[[str, Any], None]] = None,
    query: str = "",
) -> BridgePipelineResult:
    """Run Phase 2: bibcoupling → bridge graph → TDA → HDBSCAN → scoring.

    Stage names emitted via on_stage(stage, value):
      'bibcoupling_done'      → int  (new edges added)
      'domain_expansion'      → list[str]  (adjacent queries fetched)
      'single_domain_detected'→ list[str]  (unique categories; expansion failed)
      'graph_done'            → (n_nodes, n_sim_edges, n_bridge_edges)
      'tda_done'              → TDAResult
      'clustering_done'       → int  (cluster count)

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

    # Embed query for per-paper relevance scoring (used in cluster scoring + reading list)
    query_vector: Optional[np.ndarray] = None
    query_relevance_scores: dict[str, float] = {}
    if query:
        try:
            from .embedder import Embedder
            _embedder = Embedder(config.EMBED_MODEL)
            q_vecs = _embedder.encode([query], is_query=True)
            q_raw = np.array(q_vecs[0], dtype=np.float32)
            norm = np.linalg.norm(q_raw)
            query_vector = q_raw / norm if norm > 0 else q_raw
            sims = all_vectors @ query_vector
            query_relevance_scores = {
                arxiv_id: float(sims[idx]) for arxiv_id, idx in idx_map.items()
            }
        except Exception as exc:
            logger.warning("Could not embed query for relevance scoring: %s", exc)

    # ── Single-domain detection + domain expansion ────────────────────────────
    unique_cats = _primary_cats_from_papers(papers)
    if len(unique_cats) < 2:
        expanded = _try_domain_expansion(query, papers, _emit)
        if expanded:
            # Reload everything after expansion so bridge detection sees mixed-domain papers
            papers = load_papers_with_embeddings()
            idx_map = get_embedding_idx_map()
            index = faiss.read_index(str(config.FAISS_PATH))
            all_vectors = np.zeros((index.ntotal, index.d), dtype=np.float32)
            index.reconstruct_n(0, index.ntotal, all_vectors)
            if query_vector is not None:
                sims = all_vectors @ query_vector
                query_relevance_scores = {
                    arxiv_id: float(sims[idx]) for arxiv_id, idx in idx_map.items()
                }
            unique_cats = _primary_cats_from_papers(papers)

        if len(unique_cats) < 2:
            # Last resort: expansion found nothing useful
            _emit("single_domain_detected", sorted(unique_cats))
            return BridgePipelineResult(
                graph_nodes=len(papers),
                graph_edges_full=0,
                graph_edges_bridge=0,
                bibcoupling_edges_added=bibcoupling_added,
                tda_result=TDAResult(
                    enabled=False, n_cycles=0, cycles=[],
                    warning=f"Single-domain ({', '.join(sorted(unique_cats))}) — expansion found no adjacent fields",
                ),
                clusters=[],
                query_relevance_scores=query_relevance_scores,
            )

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
        query_vector=query_vector,
    )
    _emit("clustering_done", len(cluster_list))

    return BridgePipelineResult(
        graph_nodes=n_nodes,
        graph_edges_full=len(sim_edges),
        graph_edges_bridge=len(bridge_edges),
        bibcoupling_edges_added=bibcoupling_added,
        tda_result=tda_result,
        clusters=cluster_list,
        query_relevance_scores=query_relevance_scores,
    )


def run_gemma_validation(
    result: BridgePipelineResult,
    papers: dict[str, dict],
    top_n_clusters: int = 5,
    max_validate: Optional[int] = None,
    on_pair: Optional[Callable[[int, int], None]] = None,
) -> BridgePipelineResult:
    """Run Phase 3: Gemma 4 structural analogy validation on top clusters.

    Mutates result.clusters[*].validated_pairs and result.clusters[*].confidence_tier.
    Sets result.gemma_available and result.gemma_warning.

    Args:
        result: BridgePipelineResult from Phase 2 (mutated in place).
        papers: Mapping of arxiv_id → paper metadata (must include abstract).
        top_n_clusters: Validate this many top clusters.
        max_validate: Global pair budget — validate only the top-N pairs by
            cosine similarity across all clusters. None means no limit.
        on_pair: Optional callback(done, total) called after each pair.

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
        max_validate=max_validate,
        on_pair=on_pair,
    )
    return result


# ── Cross-domain categories targeted by structural channel ────────────────────
_STRUCTURAL_TOP_CATS = {"math", "nlin", "physics", "q-bio", "stat", "econ", "cond-mat"}


def run_direct_cross_domain_validation(
    result: BridgePipelineResult,
    papers: dict[str, dict],
    max_pairs: int = 10,
    on_pair: Optional[Callable[[int, int], None]] = None,
) -> BridgePipelineResult:
    """Validate cross-domain pairs via direct LLM comparison, bypassing the embedding threshold.

    Collects all (sem*, cs.*) × (str*, math.*/nlin.*/physics.*) paper combinations,
    excludes citation-connected pairs, ranks by sum of query relevance scores, and
    sends the top N to the LLM for classification.

    Mutates result.direct_cross_domain_pairs in place and returns result.
    """
    from .llm_client import call_llm
    from .db import get_citation_pairs_for_nodes

    sem_cs: list[str] = []
    str_other: list[str] = []

    for pid, p in papers.items():
        tag = p.get("query_tag", "")
        top_cat = primary_cat(p.get("categories", "[]"))
        if tag.startswith("sem") and top_cat == "cs":
            sem_cs.append(pid)
        elif tag.startswith("str") and top_cat in _STRUCTURAL_TOP_CATS:
            str_other.append(pid)

    if not sem_cs or not str_other:
        logger.debug(
            "Direct cross-domain validation skipped: sem_cs=%d str_other=%d",
            len(sem_cs), len(str_other),
        )
        return result

    citation_pairs = get_citation_pairs_for_nodes(list(papers.keys()))
    qrs = result.query_relevance_scores or {}

    candidates: list[tuple[str, str, float]] = []
    for pid_a in sem_cs:
        for pid_b in str_other:
            if frozenset({pid_a, pid_b}) in citation_pairs:
                continue
            score = qrs.get(pid_a, 0.0) + qrs.get(pid_b, 0.0)
            candidates.append((pid_a, pid_b, score))

    if not candidates:
        return result

    candidates.sort(key=lambda x: x[2], reverse=True)
    top_candidates = candidates[:max_pairs]
    total = len(top_candidates)
    validated: list[BridgePair] = []

    for i, (pid_a, pid_b, rel_sum) in enumerate(top_candidates):
        pa = papers[pid_a]
        pb = papers[pid_b]
        title_a = pa.get("title", pid_a)[:150]
        abstract_a = pa.get("abstract", "")[:600]
        title_b = pb.get("title", pid_b)[:150]
        abstract_b = pb.get("abstract", "")[:600]

        prompt = (
            "Paper A (machine learning domain):\n"
            f"Title: {title_a}\n"
            f"Abstract: {abstract_a}\n\n"
            "Paper B (mathematics/physics domain):\n"
            f"Title: {title_b}\n"
            f"Abstract: {abstract_b}\n\n"
            "Do these two papers describe the same mathematical structure or phenomenon "
            "using different domain vocabulary? Answer with one of: STRUCTURAL, "
            "METHODOLOGICAL, THEMATIC, or NONE. Then in one sentence explain why.\n\n"
            "Focus on mathematical form, not surface vocabulary. For example, "
            "'gradient descent near sharp minimum' and 'discrete map near unstable fixed "
            "point' describe the same mathematical object."
        )

        try:
            text = call_llm(prompt, timeout=30, temperature=0)
        except Exception as exc:
            logger.warning("Direct validation failed for %s ↔ %s: %s", pid_a, pid_b, exc)
            if on_pair:
                on_pair(i + 1, total)
            continue

        classification = None
        for kw in ("STRUCTURAL", "METHODOLOGICAL", "THEMATIC", "NONE"):
            if kw in text.upper():
                classification = kw
                break

        if on_pair:
            on_pair(i + 1, total)

        if not classification or classification == "NONE":
            continue

        reasoning_match = re.search(
            r"(?:STRUCTURAL|METHODOLOGICAL|THEMATIC|NONE)[.\s]*(.+)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        reasoning = reasoning_match.group(1).strip()[:400] if reasoning_match else text[:400]

        validated.append(BridgePair(
            paper_a=pid_a,
            paper_b=pid_b,
            similarity=rel_sum / 2.0,
            structure_a={"abstract_snippet": abstract_a[:200]},
            structure_b={"abstract_snippet": abstract_b[:200]},
            classification=classification,
            confidence_tier=_confidence_tier(classification, 0),
            reasoning_chain=reasoning,
            matched_properties=[],
            mismatched_properties=[],
            translation_hints=[],
            think_mode_used=False,
        ))

    # Sort: STRUCTURAL first, then METHODOLOGICAL, then THEMATIC
    validated.sort(
        key=lambda p: {"STRUCTURAL": 0, "METHODOLOGICAL": 1, "THEMATIC": 2}.get(
            p.classification, 3
        )
    )
    result.direct_cross_domain_pairs = validated
    return result
