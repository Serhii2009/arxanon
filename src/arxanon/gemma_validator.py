"""Phase 3: Gemma 4 structural analogy verification via Ollama.

Runs a two-step function-calling loop for each candidate bridge pair:
  1. extract_formal_structure — decomposes each abstract into its mathematical form
  2. classify_structural_analogy — compares the two structures and classifies the bridge

Uses think=True for Gemma 4's extended reasoning mode.  Falls back to think=False
if the thinking field is empty (some Ollama versions or quantisations may not
populate it).  The reasoning_chain in BridgePair is always non-empty.

If Ollama is unreachable or the model is not pulled, every public function
degrades gracefully: check_ollama() returns (False, msg), validate_bridge_pair()
returns None, validate_clusters() returns clusters unchanged.  No exception
propagates to callers.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from . import config
from .clusters import BridgeCluster, BridgePair, _confidence_tier, primary_cat

logger = logging.getLogger(__name__)

# ── Ollama tool schemas ───────────────────────────────────────────────────────

EXTRACT_STRUCTURE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "extract_formal_structure",
        "description": (
            "Extract the formal mathematical or computational structure "
            "from a scientific paper abstract."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "problem_type": {
                    "type": "string",
                    "description": "Type of mathematical problem, e.g. stability_analysis, optimization, estimation.",
                },
                "key_relationships": {
                    "type": "string",
                    "description": "Key equations, inequalities, or formal relationships described.",
                },
                "solution_approach": {
                    "type": "string",
                    "description": "Primary technique, method, or proof strategy used.",
                },
                "assumptions": {
                    "type": "string",
                    "description": "Key assumptions, constraints, or preconditions.",
                },
                "domain_vocabulary": {
                    "type": "string",
                    "description": "Domain-specific vocabulary that may differ from equivalent concepts in other fields.",
                },
            },
            "required": [
                "problem_type",
                "key_relationships",
                "solution_approach",
                "assumptions",
                "domain_vocabulary",
            ],
        },
    },
}

CLASSIFY_BRIDGE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "classify_structural_analogy",
        "description": "Classify the type of structural analogy between two papers' formal structures.",
        "parameters": {
            "type": "object",
            "properties": {
                "classification": {
                    "type": "string",
                    "enum": ["STRUCTURAL", "METHODOLOGICAL", "THEMATIC", "SUPERFICIAL"],
                    "description": (
                        "STRUCTURAL=same mathematical form in different domains; "
                        "METHODOLOGICAL=same technique applied differently; "
                        "THEMATIC=same research theme but different formalism; "
                        "SUPERFICIAL=shared vocabulary without structural correspondence."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step reasoning chain explaining the classification.",
                },
                "matched_properties": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Formal properties that correspond between the two papers.",
                },
                "mismatched_properties": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Properties that differ or don't translate between the papers.",
                },
                "translation_hints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "term_a": {"type": "string"},
                            "term_b": {"type": "string"},
                        },
                    },
                    "description": "Pairs of equivalent terms: what paper A calls term_a, paper B calls term_b.",
                },
            },
            "required": [
                "classification",
                "reasoning",
                "matched_properties",
                "mismatched_properties",
            ],
        },
    },
}


# ── Connectivity check ────────────────────────────────────────────────────────

def check_ollama(
    base_url: str = config.OLLAMA_BASE_URL,
    model: str = config.GEMMA_MODEL,
) -> tuple[bool, Optional[str]]:
    """Check whether Ollama is running and the requested model is available.

    Returns:
        (True, None) if ready.
        (False, error_message) otherwise — never raises.
    """
    import requests  # already a hard dep

    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        resp.raise_for_status()
        data = resp.json()
        model_names: list[str] = [m.get("name", "") for m in data.get("models", [])]
        model_base = model.split(":")[0]
        if not any(name.startswith(model_base) for name in model_names):
            return False, (
                f"Model '{model}' not found in Ollama.  "
                f"Run: ollama pull {model}"
            )
        return True, None
    except requests.exceptions.ConnectionError:
        return False, f"Ollama not running at {base_url}.  Run: ollama serve"
    except requests.exceptions.Timeout:
        return False, f"Ollama connection timed out at {base_url}"
    except Exception as exc:
        return False, f"Ollama check failed: {exc}"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _chat_with_think_fallback(
    model: str,
    messages: list[dict],
    tools: list[dict],
    timeout: int,
    base_url: str = config.OLLAMA_BASE_URL,
) -> tuple[Any, bool]:
    """Call ollama.chat with think=True; fall back to think=False if reasoning is empty.

    Returns:
        (response, think_mode_used) where think_mode_used is True only when
        response.message.thinking was non-empty.
    """
    import ollama as ollama_lib  # guarded: caller guarantees Ollama is available

    client = ollama_lib.Client(host=base_url, timeout=timeout)

    try:
        response = client.chat(
            model=model,
            messages=messages,
            tools=tools,
            think=True,
            options={"temperature": 0},
        )
        thinking = getattr(response.message, "thinking", None)
        if thinking:
            return response, True
        logger.info("think=True produced empty thinking field; retrying with think=False")
    except Exception as exc:
        logger.warning("think=True call failed (%s); retrying with think=False", exc)

    response = client.chat(
        model=model,
        messages=messages,
        tools=tools,
        think=False,
        options={"temperature": 0},
    )
    return response, False


# ── Structure extraction ──────────────────────────────────────────────────────

def extract_formal_structure(
    abstract: str,
    model: str = config.GEMMA_MODEL,
    base_url: str = config.OLLAMA_BASE_URL,
    timeout: int = 60,
) -> Optional[dict]:
    """Extract the formal mathematical structure from a paper abstract.

    Returns:
        Dict with keys problem_type, key_relationships, solution_approach,
        assumptions, domain_vocabulary — or None on failure.
    """
    try:
        import ollama as ollama_lib
    except ImportError:
        logger.error("ollama package not installed.  Run: pip install ollama")
        return None

    client = ollama_lib.Client(host=base_url, timeout=timeout)
    messages = [
        {
            "role": "user",
            "content": (
                "Extract the formal mathematical structure from this paper abstract.  "
                "Call the extract_formal_structure function.\n\n"
                f"Abstract:\n{abstract[:2000]}"
            ),
        }
    ]

    try:
        response = client.chat(
            model=model,
            messages=messages,
            tools=[EXTRACT_STRUCTURE_TOOL],
            options={"temperature": 0},
        )
    except Exception as exc:
        logger.warning("extract_formal_structure failed: %s", exc)
        return None

    if not response.message.tool_calls:
        return None

    try:
        args = response.message.tool_calls[0].function.arguments
        return dict(args) if isinstance(args, dict) else {}
    except Exception as exc:
        logger.warning("Failed to parse structure extraction response: %s", exc)
        return None


# ── Bridge pair validation ────────────────────────────────────────────────────

def validate_bridge_pair(
    paper_a: dict,
    paper_b: dict,
    sim_score: float,
    model: str = config.GEMMA_MODEL,
    base_url: str = config.OLLAMA_BASE_URL,
    timeout: int = 120,
) -> Optional[BridgePair]:
    """Validate and classify the structural analogy between two papers.

    Steps:
      1. extract_formal_structure for paper_a
      2. extract_formal_structure for paper_b
      3. classify_structural_analogy (with think=True fallback)

    Returns:
        BridgePair if classification is not SUPERFICIAL, None otherwise.
        Also returns None on any Ollama failure (never raises).
    """
    half_timeout = max(timeout // 2, 30)

    structure_a = extract_formal_structure(
        paper_a.get("abstract", ""), model=model, base_url=base_url, timeout=half_timeout
    )
    if structure_a is None:
        return None

    structure_b = extract_formal_structure(
        paper_b.get("abstract", ""), model=model, base_url=base_url, timeout=half_timeout
    )
    if structure_b is None:
        return None

    cat_a = paper_a.get("categories", "[]")
    cat_b = paper_b.get("categories", "[]")
    messages = [
        {
            "role": "user",
            "content": (
                "Compare the formal mathematical structures of these two papers and classify "
                "their analogy.  Call classify_structural_analogy.\n\n"
                f"Paper A ({paper_a.get('arxiv_id', '?')}, category: {cat_a}):\n"
                f"{json.dumps(structure_a)}\n\n"
                f"Paper B ({paper_b.get('arxiv_id', '?')}, category: {cat_b}):\n"
                f"{json.dumps(structure_b)}"
            ),
        }
    ]

    try:
        response, think_mode_used = _chat_with_think_fallback(
            model=model,
            messages=messages,
            tools=[CLASSIFY_BRIDGE_TOOL],
            timeout=timeout,
            base_url=base_url,
        )
    except Exception as exc:
        logger.warning("Bridge classification failed: %s", exc)
        return None

    if not response.message.tool_calls:
        return None

    try:
        args = response.message.tool_calls[0].function.arguments
        if not isinstance(args, dict):
            return None
    except Exception as exc:
        logger.warning("Failed to parse classification response: %s", exc)
        return None

    classification = args.get("classification", "SUPERFICIAL")
    if classification == "SUPERFICIAL":
        return None

    # Reasoning chain: prefer thinking trace, then content, then args field
    thinking = getattr(response.message, "thinking", None)
    if think_mode_used and thinking:
        reasoning = thinking
    else:
        reasoning = response.message.content or args.get("reasoning", "")
        if not reasoning:
            reasoning = args.get("reasoning", "[No reasoning captured]")

    n_matched = len(args.get("matched_properties", []))
    return BridgePair(
        paper_a=paper_a.get("arxiv_id", "?"),
        paper_b=paper_b.get("arxiv_id", "?"),
        similarity=sim_score,
        structure_a=structure_a,
        structure_b=structure_b,
        classification=classification,
        confidence_tier=_confidence_tier(classification, n_matched),
        reasoning_chain=reasoning,
        matched_properties=args.get("matched_properties", []),
        mismatched_properties=args.get("mismatched_properties", []),
        translation_hints=args.get("translation_hints", []),
        think_mode_used=think_mode_used,
    )


# ── Cluster validation ────────────────────────────────────────────────────────

def _get_cross_domain_pairs(
    cluster: BridgeCluster,
    papers: dict[str, dict],
) -> list[tuple[str, str, float]]:
    """Return cross-domain bridge edges, sorted by similarity descending."""
    result: list[tuple[str, str, float]] = []
    for id_a, id_b, sim in cluster.bridge_edges:
        pa = papers.get(id_a, {})
        pb = papers.get(id_b, {})
        cat_a = primary_cat(pa.get("categories", "[]"))
        cat_b = primary_cat(pb.get("categories", "[]"))
        if cat_a != cat_b:
            result.append((id_a, id_b, sim))
    result.sort(key=lambda x: x[2], reverse=True)
    return result


def validate_clusters(
    clusters: list[BridgeCluster],
    papers: dict[str, dict],
    top_n_clusters: int = 5,
    max_pairs_per_cluster: int = 20,
    on_pair: Optional[Callable[[int, int], None]] = None,
) -> list[BridgeCluster]:
    """Run Gemma 4 bridge validation on the top-N clusters.

    Mutates cluster.validated_pairs and cluster.confidence_tier in place.

    Args:
        clusters: Full ranked cluster list (sorted by composite score).
        papers: Mapping of arxiv_id → paper metadata (must include abstract).
        top_n_clusters: Validate this many top clusters.
        max_pairs_per_cluster: Validate at most this many pairs per cluster.
        on_pair: Optional callback(done, total) called after each pair.

    Returns:
        Updated clusters list (same objects, mutated in place).
    """
    top = clusters[:top_n_clusters]

    # Pre-compute cross-domain pairs for each cluster
    pairs_per_cluster = [
        _get_cross_domain_pairs(c, papers)[:max_pairs_per_cluster]
        for c in top
    ]
    total_to_validate = sum(len(p) for p in pairs_per_cluster)
    total_done = 0

    for cluster, pairs in zip(top, pairs_per_cluster):
        for id_a, id_b, sim in pairs:
            pa = papers.get(id_a)
            pb = papers.get(id_b)
            if not pa or not pb:
                continue

            bridge_pair = validate_bridge_pair(
                paper_a=pa,
                paper_b=pb,
                sim_score=sim,
            )
            total_done += 1
            if on_pair:
                on_pair(total_done, total_to_validate)

            if bridge_pair is not None:
                cluster.validated_pairs.append(bridge_pair)

        if cluster.validated_pairs:
            # Sort: STRUCTURAL first, then by n_matched desc
            cluster.validated_pairs.sort(
                key=lambda p: (
                    {"STRUCTURAL": 0, "METHODOLOGICAL": 1, "THEMATIC": 2}.get(
                        p.classification, 3
                    ),
                    -len(p.matched_properties),
                )
            )
            cluster.confidence_tier = cluster.validated_pairs[0].confidence_tier

    return clusters
