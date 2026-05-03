"""Phase 3: Gemma 4 structural analogy verification via Ollama.

Two validation modes, controlled by config.GEMMA_SIMPLE_MODE (default: True):

Simple mode (default): single plain-text call per pair — no function calling,
no think=True.  Works with small models (gemma4:e2b, 2b) on CPU in <20s/pair.

Agentic mode (ARXANON_GEMMA_SIMPLE=0): two-step function-calling loop:
  1. extract_formal_structure — decomposes each abstract into its mathematical form
  2. classify_structural_analogy — compares the two structures and classifies the bridge
  Uses think=True for Gemma 4's extended reasoning mode with fallback to think=False.

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
    if config.USE_OPENROUTER:
        return True, None

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


# ── Simple single-call validation ────────────────────────────────────────────

def _parse_simple_response(text: str) -> dict:
    """Parse the 3-line structured response from the simple validation prompt.

    Falls back to keyword scan if the model doesn't follow the format.
    Returns dict with keys: classification, reasoning, matched (list[str]).
    """
    import re

    classification = None
    reasoning = ""
    matched: list[str] = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if m := re.match(r"CLASSIFICATION\s*:\s*(\w+)", line, re.IGNORECASE):
            classification = m.group(1).upper()
        elif m := re.match(r"REASON\s*:\s*(.+)", line, re.IGNORECASE):
            reasoning = m.group(1).strip()
        elif m := re.match(r"MATCHED\s*:\s*(.+)", line, re.IGNORECASE):
            matched = [s.strip() for s in m.group(1).split(",") if s.strip()]

    valid = {"STRUCTURAL", "METHODOLOGICAL", "THEMATIC", "SUPERFICIAL"}
    if classification not in valid:
        upper = text.upper()
        for kw in ("STRUCTURAL", "METHODOLOGICAL", "THEMATIC", "SUPERFICIAL"):
            if kw in upper:
                classification = kw
                break
        else:
            classification = "SUPERFICIAL"

    return {"classification": classification, "reasoning": reasoning, "matched": matched}


def _validate_bridge_pair_simple(
    paper_a: dict,
    paper_b: dict,
    sim_score: float,
    model: str = config.GEMMA_MODEL,
    base_url: str = config.OLLAMA_BASE_URL,
    timeout: int = 20,
) -> Optional["BridgePair"]:
    """Single-call plain-text validation — no function calling, no think mode.

    Designed for small models (gemma4:e2b, 2b) on CPU.  Completes in <20s/pair.
    Returns None for SUPERFICIAL classification or on any LLM failure.
    """
    cat_a = paper_a.get("categories", "[]")
    cat_b = paper_b.get("categories", "[]")
    abstract_a = paper_a.get("abstract", "")[:600]
    abstract_b = paper_b.get("abstract", "")[:600]

    prompt = (
        "Compare these two scientific papers and classify their structural relationship.\n\n"
        f"Paper A [{cat_a}]:\n{abstract_a}\n\n"
        f"Paper B [{cat_b}]:\n{abstract_b}\n\n"
        "STRUCTURAL = same mathematical/algorithmic form in different fields\n"
        "METHODOLOGICAL = same technique applied to different problems\n"
        "THEMATIC = related topic but different formalism\n"
        "SUPERFICIAL = surface keyword overlap only\n\n"
        "Reply with exactly these 3 lines:\n"
        "CLASSIFICATION: [STRUCTURAL|METHODOLOGICAL|THEMATIC|SUPERFICIAL]\n"
        "REASON: [one sentence]\n"
        "MATCHED: [property1, property2, property3]"
    )

    try:
        from .llm_client import call_llm
        text = call_llm(prompt, timeout=timeout, temperature=0)
    except Exception as exc:
        logger.warning("simple validation failed: %s", exc)
        return None
    parsed = _parse_simple_response(text)

    classification = parsed["classification"]
    if classification == "SUPERFICIAL":
        return None

    matched = parsed["matched"]
    reasoning = parsed["reasoning"] or text[:300] or "[No reasoning captured]"

    return BridgePair(
        paper_a=paper_a.get("arxiv_id", "?"),
        paper_b=paper_b.get("arxiv_id", "?"),
        similarity=sim_score,
        structure_a={"abstract_snippet": paper_a.get("abstract", "")[:200]},
        structure_b={"abstract_snippet": paper_b.get("abstract", "")[:200]},
        classification=classification,
        confidence_tier=_confidence_tier(classification, len(matched)),
        reasoning_chain=reasoning,
        matched_properties=matched,
        mismatched_properties=[],
        translation_hints=[],
        think_mode_used=False,
    )


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
    if config.GEMMA_SIMPLE_MODE:
        return _validate_bridge_pair_simple(paper_a, paper_b, sim_score, model, base_url, timeout)

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

def _retrieval_channel(query_tag: str) -> str:
    """Map a query_tag to 'semantic', 'structural', or 'unknown'."""
    if query_tag.startswith("sem"):
        return "semantic"
    if query_tag.startswith("str"):
        return "structural"
    return "unknown"


def _get_cross_domain_pairs(
    cluster: BridgeCluster,
    papers: dict[str, dict],
) -> list[tuple[str, str, float]]:
    """Return bridge edges where papers come from different retrieval channels AND categories.

    When dual-channel tags (sem*/str*) are present, enforces that one paper comes from the
    semantic channel and the other from the structural channel, in addition to requiring
    different top-level arXiv categories.

    Falls back to category-only filtering for legacy single-channel tags (q*, adj*).
    """
    result: list[tuple[str, str, float]] = []
    for id_a, id_b, sim in cluster.bridge_edges:
        pa = papers.get(id_a, {})
        pb = papers.get(id_b, {})
        tag_a = pa.get("query_tag", "")
        tag_b = pb.get("query_tag", "")
        cat_a = primary_cat(pa.get("categories", "[]"))
        cat_b = primary_cat(pb.get("categories", "[]"))

        ch_a = _retrieval_channel(tag_a)
        ch_b = _retrieval_channel(tag_b)

        if ch_a != "unknown" and ch_b != "unknown":
            # Dual-channel mode: require one sem and one str, plus different top-level cats
            if ch_a != ch_b and cat_a != cat_b:
                result.append((id_a, id_b, sim))
        else:
            # Legacy / single-channel: fall back to category-only check
            if cat_a != cat_b:
                result.append((id_a, id_b, sim))

    result.sort(key=lambda x: x[2], reverse=True)
    return result


def validate_clusters(
    clusters: list[BridgeCluster],
    papers: dict[str, dict],
    top_n_clusters: int = 5,
    max_pairs_per_cluster: int = 20,
    max_validate: Optional[int] = None,
    on_pair: Optional[Callable[[int, int], None]] = None,
) -> list[BridgeCluster]:
    """Run Gemma 4 bridge validation on the top-N clusters.

    Mutates cluster.validated_pairs and cluster.confidence_tier in place.

    Args:
        clusters: Full ranked cluster list (sorted by composite score).
        papers: Mapping of arxiv_id → paper metadata (must include abstract).
        top_n_clusters: Validate this many top clusters.
        max_pairs_per_cluster: Validate at most this many pairs per cluster
            (used only when max_validate is None).
        max_validate: Global budget — validate only the top-N pairs by cosine
            similarity across all clusters combined. When set, overrides
            max_pairs_per_cluster. Use for testing with lightweight models.
        on_pair: Optional callback(done, total) called after each pair.

    Returns:
        Updated clusters list (same objects, mutated in place).
    """
    top = clusters[:top_n_clusters]

    if max_validate is not None:
        # Global top-N: merge all cross-domain pairs, sort by sim desc, take top N
        tagged: list[tuple[int, str, str, float]] = [
            (ci, a, b, s)
            for ci, c in enumerate(top)
            for a, b, s in _get_cross_domain_pairs(c, papers)
        ]
        tagged.sort(key=lambda x: x[3], reverse=True)
        per_cluster: list[list[tuple[str, str, float]]] = [[] for _ in top]
        for ci, a, b, s in tagged[:max_validate]:
            per_cluster[ci].append((a, b, s))
        pairs_per_cluster = per_cluster
    else:
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
