"""Phase 4 output writer: saves per-session files to ./{query_slug}/."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from . import config
from .clusters import BridgeCluster, BridgePair, BridgePipelineResult

logger = logging.getLogger(__name__)

_CONNECTION_LABELS = {
    "STRUCTURAL":     "shares the same mathematical mechanism",
    "METHODOLOGICAL": "uses the same analytical approach",
    "THEMATIC":       "addresses the same phenomenon from a different angle",
}

_FIELD_DISTANCE_FROM_CS: dict[str, int] = {
    "cs":       0,
    "stat":     1,
    "math":     2,
    "eess":     2,
    "physics":  3,
    "cond-mat": 3,
    "nlin":     3,
    "q-bio":    4,
    "econ":     4,
    "q-fin":    4,
}

_CS_NOISE_SUBCATS = {"cs.HC", "cs.RO", "cs.CR", "cs.MA"}


def _is_cs_subfield_noise(pair: "BridgePair", papers: dict) -> bool:
    pb = papers.get(pair.paper_b, {})
    try:
        cats = json.loads(pb.get("categories", "[]") or "[]")
        return bool(cats) and cats[0] in _CS_NOISE_SUBCATS
    except Exception:
        return False


def _reasoning_is_surface_only(pair: "BridgePair") -> bool:
    r = (pair.reasoning_chain or "").lower()
    return (
        "do not share a common mathematical mechanism" in r
        or "they only share the surface-level term" in r
    )


def _field_dist(pair: "BridgePair", papers: dict) -> int:
    """Return field distance of paper_b from CS (higher = more distant)."""
    pb = papers.get(pair.paper_b, {})
    try:
        cats = json.loads(pb.get("categories", "[]") or "[]")
        top = cats[0].split(".")[0] if cats else "?"
    except Exception:
        top = "?"
    return _FIELD_DISTANCE_FROM_CS.get(top, 2)


# ── Public entry point ────────────────────────────────────────────────────────

def _query_slug(query: str) -> str:
    words = re.sub(r"[^a-z0-9 ]", "", query.lower()).split()[:4]
    return "_".join(words) if words else "arxanon_session"


def save_session(
    query: str,
    bridge_result: BridgePipelineResult,
    papers: dict[str, dict],
) -> tuple[Path, Optional[str]]:
    """Save all output files to ./{query_slug}/. Returns (directory path, synthesis text)."""
    slug = _query_slug(query)
    out_dir = Path(slug)
    if out_dir.exists():
        i = 2
        while Path(f"{slug}_{i}").exists():
            i += 1
        out_dir = Path(f"{slug}_{i}")
    out_dir.mkdir(exist_ok=True)

    top_clusters = bridge_result.clusters[:5]

    directions = _write_bridge_report(out_dir, query, bridge_result, papers, top_clusters)
    _write_sources_bib(out_dir, top_clusters, papers)
    _write_bridge_map(out_dir, top_clusters, papers, bridge_result.direct_cross_domain_pairs or [])

    return out_dir, directions


# ── Gemma research directions ─────────────────────────────────────────────────

def _gemma_research_directions(
    query: str,
    top_papers_text: str,
    bridge_pairs_text: str,
    direct_pairs: list | None = None,
    papers: dict | None = None,
    str_papers_text: str = "",
    structural_queries: list[str] | None = None,
    query_relevance_scores: dict[str, float] | None = None,
) -> Optional[str]:
    try:
        from .llm_client import call_llm

        # Build structural-queries preamble
        sq_text = ""
        if structural_queries:
            sq_lines = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(structural_queries))
            sq_text = (
                f"The system translated the ML phenomenon into outside-ML vocabulary "
                f"and searched arXiv with these structural queries:\n{sq_lines}\n\n"
            )

        # Build enriched bridge context from top non-NONE pairs (abstracts + reasoning)
        # Most-distant field first so the LLM sees the novel outside-ML paper at CONNECTION 1
        _papers_ref = papers or {}
        top_pairs = sorted(
            (p for p in (direct_pairs or []) if p.classification != "NONE"),
            key=lambda p: (
                -_field_dist(p, _papers_ref),
                {"STRUCTURAL": 0, "METHODOLOGICAL": 1, "THEMATIC": 2}.get(p.classification, 3),
            ),
        )[:3]
        bridge_rich_lines: list[str] = []
        for i, pair in enumerate(top_pairs, 1):
            pa = (papers or {}).get(pair.paper_a, {})
            pb = (papers or {}).get(pair.paper_b, {})
            try:
                cats_b = json.loads(pb.get("categories", "[]") or "[]")
                cat_b = cats_b[0] if cats_b else "?"
            except Exception:
                cat_b = "?"
            _abs_a = pa.get("abstract") or ""
            _abs_b = pb.get("abstract") or ""
            bridge_rich_lines += [
                f"CONNECTION {i}: {pa.get('title', pair.paper_a)} (ML) "
                f"↔ {pb.get('title', pair.paper_b)} ({cat_b})",
                f"ML paper (arxiv:{pair.paper_a}):",
                f"  {(_abs_a[:600] + '…') if len(_abs_a) > 600 else _abs_a}",
                f"Outside-ML paper (arxiv:{pair.paper_b}, {cat_b}):",
                f"  {(_abs_b[:600] + '…') if len(_abs_b) > 600 else _abs_b}",
                f"Why they connect: {pair.reasoning_chain}",
                "",
            ]
        bridge_context = "\n".join(bridge_rich_lines) if bridge_rich_lines else bridge_pairs_text

        if bridge_rich_lines:
            prompt = (
                f'You are writing a research briefing for an ML researcher.\n\n'
                f'Their question: "{query}"\n\n'
                f"{sq_text}"
                f"The following cross-domain connections were validated:\n\n"
                f"{bridge_context}\n"
                f"Background ML context:\n{top_papers_text}\n\n"
                "Write EXACTLY two sections:\n\n"
                "**What we found:** (2-3 sentences) Lead with the most interesting outside-ML paper — "
                "name the field, what that paper found about this phenomenon, and why it matters for "
                "the researcher's question. Plain English. No classification labels.\n\n"
                "**Experiment:** Derive the experiment from the outside-ML paper that suggests the "
                "most ACTIONABLE and NOVEL prediction — one that:\n"
                "(a) has not already been published in an ML paper\n"
                "(b) can be tested by varying something in a standard training run (learning rate "
                "schedule, compute budget, model size trajectory) without requiring specialized "
                "measurement tools\n"
                "(c) makes a specific quantitative prediction (not just 'measure if X correlates with Y')\n\n"
                "Prefer experiments a research lab could run next week over experiments requiring "
                "months of infrastructure. Do not propose measuring fractal dimensions, activation "
                "manifold topology, or other quantities requiring custom tooling.\n\n"
                "Format as: '[Outside-ML mechanism] predicts [specific quantitative outcome]. "
                "Test: [exact procedure a lab could follow].'\n\n"
                "Tag every factual claim inline:\n"
                "  [GROUNDED: arxiv:XXXX] — directly stated in an abstract above\n"
                "  [INFERRED] — logical deduction from the papers\n"
                "  [SPECULATIVE] — goes beyond the papers\n\n"
                "Hard constraint: never use [GROUNDED: arxiv:X] for a claim not directly in "
                "the abstract text above. When uncertain, use [INFERRED].\n"
                "Ground every claim in the actual papers listed. Do not invent results."
            )
        else:
            str_section = (
                "\nRELEVANT OUTSIDE-ML PAPERS (structural search channel — unvalidated):\n"
                f"{str_papers_text}\n"
                if str_papers_text else ""
            )
            prompt = (
                f'Research query: "{query}"\n\n'
                f"{sq_text}"
                f"Top papers by relevance:\n{top_papers_text}\n"
                f"{str_section}\n"
                f"Validated bridge connections:\n{bridge_pairs_text}\n\n"
                "Based only on what is listed above, write two short sections:\n\n"
                "**What we found:** (2-3 sentences) What do the most relevant papers show about this "
                "phenomenon? If any outside-ML paper appears, name its field and what it found.\n\n"
                "**Experiment:** (1-2 sentences) One specific experiment the researcher could run.\n\n"
                "Tag every factual claim with [GROUNDED: arxiv:XXXX], [INFERRED], or [SPECULATIVE].\n"
                "Ground every claim in the actual papers listed. Do not invent results."
            )

        content = call_llm(prompt, timeout=45, temperature=0.3)
        return content.strip() if content.strip() else None
    except Exception as exc:
        logger.debug("Gemma research directions failed: %s", exc)
        return None


# ── Deterministic fallback when LLM synthesis is unavailable ─────────────────

def _deterministic_directions_fallback(query: str, pairs: list, papers: dict) -> list[str]:
    n = len(pairs)
    out = [
        "**Cross-Domain Finding (deterministic — LLM synthesis unavailable):**",
        f"{n} structural connection(s) found between your ML papers and outside-ML literature.",
        "",
    ]
    if pairs:
        best = pairs[0]
        pa = papers.get(best.paper_a, {})
        pb = papers.get(best.paper_b, {})
        try:
            cats_b = json.loads(pb.get("categories", "[]") or "[]")
            cat_b = cats_b[0] if cats_b else "?"
        except Exception:
            cat_b = "?"
        out += [
            "Strongest connection:",
            f"ML paper: \"{pa.get('title', best.paper_a)}\" (arxiv:{best.paper_a})",
            f"Outside-ML paper: \"{pb.get('title', best.paper_b)}\" (arxiv:{best.paper_b}, {cat_b})",
            f"LLM validation reasoning: {best.reasoning_chain or '[none]'}",
            "",
            "To generate full synthesis, ensure OpenRouter API is accessible.",
        ]
    else:
        out.append("No validated pairs available. Re-run with more papers or a different query.")
    return out


# ── bridge_report.md ──────────────────────────────────────────────────────────

def _append_pair_entry(lines: list[str], pair: "BridgePair", papers: dict) -> None:
    pa = papers.get(pair.paper_a, {})
    pb = papers.get(pair.paper_b, {})
    title_a = pa.get("title", pair.paper_a)
    title_b = pb.get("title", pair.paper_b)
    try:
        cat_b_list = json.loads(pb.get("categories", "[]") or "[]")
        cat_b_label = cat_b_list[0] if cat_b_list else "?"
    except Exception:
        cat_b_label = "?"
    conn_label = _CONNECTION_LABELS.get(pair.classification, pair.classification)
    reasoning = ""
    if pair.reasoning_chain and pair.reasoning_chain != "[No reasoning captured]":
        reasoning = pair.reasoning_chain.replace("\n", " ")
    lines += [f"- **{title_b}** (arxiv:{pair.paper_b}, {cat_b_label})"]
    lines.append(f"  {conn_label}.")
    if reasoning:
        lines.append(f"  {reasoning}")
    lines += [f"  → Relates to: **{title_a}** (arxiv:{pair.paper_a})", ""]


def _write_grouped_connections(lines: list[str], pairs: list["BridgePair"], papers: dict) -> None:
    """Group pairs by outside-ML paper (paper_b); one block per unique framework."""
    groups: dict[str, list["BridgePair"]] = {}
    for pair in pairs:
        groups.setdefault(pair.paper_b, []).append(pair)

    for pid_b, group_pairs in groups.items():
        pb = papers.get(pid_b, {})
        title_b = pb.get("title", pid_b)
        try:
            cat_b_list = json.loads(pb.get("categories", "[]") or "[]")
            cat_b_label = cat_b_list[0] if cat_b_list else "?"
        except Exception:
            cat_b_label = "?"

        best = group_pairs[0]
        conn_label = _CONNECTION_LABELS.get(best.classification, best.classification)

        lines += [f"**{title_b}** (arxiv:{pid_b}, {cat_b_label})"]
        lines.append(f"{conn_label}.")

        reasoning = ""
        if best.reasoning_chain and best.reasoning_chain != "[No reasoning captured]":
            reasoning = best.reasoning_chain.replace("\n", " ")
        if reasoning:
            lines += [reasoning, ""]
        else:
            lines.append("")

        n = len(group_pairs)
        lines.append(f"Connects to {n} paper{'s' if n != 1 else ''} in your field:")
        for pair in group_pairs:
            pa = papers.get(pair.paper_a, {})
            title_a = pa.get("title", pair.paper_a)
            ml_reason = ""
            if pair.reasoning_chain and pair.reasoning_chain != "[No reasoning captured]":
                chain = pair.reasoning_chain.replace("\n", " ")
                dot_idx = chain.find(". ")
                ml_reason = chain[:dot_idx + 1] if dot_idx > 0 else chain[:300]
            if ml_reason:
                lines.append(f"- **{title_a}** (arxiv:{pair.paper_a}): {ml_reason}")
            else:
                lines.append(f"- **{title_a}** (arxiv:{pair.paper_a})")
        lines.append("")


def _write_bridge_report(
    out_dir: Path,
    query: str,
    result: BridgePipelineResult,
    papers: dict,
    top_clusters: list[BridgeCluster],
) -> Optional[str]:
    qrs = result.query_relevance_scores or {}
    direct_pairs = result.direct_cross_domain_pairs or []
    all_validated = [p for c in top_clusters for p in c.validated_pairs]

    # ── All non-NONE pairs, STRUCTURAL first ──────────────────────────────────
    strong_pairs = sorted(
        (p for p in direct_pairs if p.classification == "STRUCTURAL"),
        key=lambda p: p.similarity, reverse=True,
    )
    related_pairs = sorted(
        (p for p in direct_pairs if p.classification in ("METHODOLOGICAL", "THEMATIC")),
        key=lambda p: p.similarity, reverse=True,
    )

    # Fix 2: each paper_b appears in exactly one section — strong wins
    strong_paper_bs = {p.paper_b for p in strong_pairs}
    extra = sorted(
        [p for p in related_pairs if p.paper_b in strong_paper_bs],
        key=lambda p: p.similarity, reverse=True,
    )
    strong_pairs = strong_pairs + extra
    related_pairs = [p for p in related_pairs if p.paper_b not in strong_paper_bs]

    # Fix 3: filter CS-subfield noise and surface-level vocabulary matches from related only
    related_pairs = [
        p for p in related_pairs
        if not _is_cs_subfield_noise(p, papers)
        and not _reasoning_is_surface_only(p)
    ]

    all_display_pairs = strong_pairs + related_pairs
    combined_pairs = all_display_pairs if all_display_pairs else all_validated

    # ── Build synthesis context ───────────────────────────────────────────────
    top5 = sorted(qrs, key=qrs.__getitem__, reverse=True)[:5]
    top_papers_lines: list[str] = []
    for pid in top5:
        p = papers.get(pid, {})
        title = p.get("title", pid)
        _abs = p.get("abstract") or ""
        abstract = (_abs[:600] + "…") if len(_abs) > 600 else _abs
        top_papers_lines.append(f"- {title} (arxiv:{pid}): {abstract}")
    top_papers_text = "\n".join(top_papers_lines) if top_papers_lines else "No papers."

    str_pids = [pid for pid, p in papers.items()
                if (p.get("query_tag") or "").startswith("str")]
    str_by_qrs = sorted(str_pids, key=lambda p: qrs.get(p, 0.0), reverse=True)[:5]
    str_papers_lines: list[str] = []
    for pid in str_by_qrs:
        p = papers.get(pid, {})
        title = p.get("title", pid)
        _abs = p.get("abstract") or ""
        abstract = (_abs[:600] + "…") if len(_abs) > 600 else _abs
        try:
            cats = json.loads(p.get("categories", "[]") or "[]")
            cat = cats[0] if cats else "?"
        except Exception:
            cat = "?"
        str_papers_lines.append(f"- {title} (arxiv:{pid}, {cat}): {abstract}")
    str_papers_text = "\n".join(str_papers_lines)

    if combined_pairs:
        ctx_lines: list[str] = []
        for pair in combined_pairs[:5]:
            pa = papers.get(pair.paper_a, {})
            pb = papers.get(pair.paper_b, {})
            ta = pa.get("title", pair.paper_a)
            tb = pb.get("title", pair.paper_b)
            ctx_lines.append(
                f"- {pair.classification}: arxiv:{pair.paper_a} ({ta}) "
                f"↔ arxiv:{pair.paper_b} ({tb})"
            )
        bridge_pairs_text = "\n".join(ctx_lines)
    else:
        bridge_pairs_text = "None found."

    # ── Run synthesis LLM first (result goes at top of report) ───────────────
    directions = _gemma_research_directions(
        query, top_papers_text, bridge_pairs_text,
        direct_pairs=direct_pairs, papers=papers,
        str_papers_text=str_papers_text,
        structural_queries=result.structural_queries,
        query_relevance_scores=result.query_relevance_scores,
    )
    n_grounded = len(re.findall(r'\[GROUNDED:', directions)) if directions else 0
    n_inferred = len(re.findall(r'\[INFERRED\]', directions)) if directions else 0
    n_speculative = len(re.findall(r'\[SPECULATIVE\]', directions)) if directions else 0

    # ── Assemble report ───────────────────────────────────────────────────────
    lines: list[str] = [f"# Research Analysis: {query}", ""]

    # Section 1: Synthesis at top
    if directions:
        lines += [directions, ""]
    else:
        lines += _deterministic_directions_fallback(query, combined_pairs, papers)
        lines.append("")

    # Section 2: Cross-Domain Connections
    lines += ["## Cross-Domain Connections", ""]

    if all_display_pairs:
        if strong_pairs:
            n_str_fw = len({p.paper_b for p in strong_pairs})
            lines += [f"### Strong connections ({n_str_fw})", ""]
            _write_grouped_connections(lines, strong_pairs, papers)
        if related_pairs:
            n_rel_fw = len({p.paper_b for p in related_pairs})
            lines += [f"### Related connections ({n_rel_fw})", ""]
            _write_grouped_connections(lines, related_pairs, papers)
    elif all_validated:
        lines.append("*Connections found via embedding similarity (no direct LLM validation):*\n")
        for pair in all_validated:
            _append_pair_entry(lines, pair, papers)
    else:
        str_cats: list[str] = []
        for _p in papers.values():
            if (_p.get("query_tag") or "").startswith("str"):
                try:
                    _cats = json.loads(_p.get("categories", "[]") or "[]")
                    if _cats:
                        str_cats.append(_cats[0])
                except Exception:
                    pass
        str_cats = sorted(set(str_cats))
        if str_cats:
            lines += [
                "No cross-domain connections were validated in this run.",
                f"The structural channel retrieved papers from: {', '.join(str_cats)}.",
                "Consider rephrasing the query to emphasize the mathematical or "
                "mechanistic aspect of the phenomenon.",
                "",
            ]
        else:
            lines += [
                "No cross-domain connections were validated in this run.",
                "The structural channel found no papers. Consider rephrasing to "
                "emphasize the mathematical or mechanistic aspect of the phenomenon.",
                "",
            ]

    # Section 3: Papers Retrieved (de-cluttered — no relevance scores)
    lines += ["## Papers Retrieved", ""]

    ranked = sorted(qrs, key=qrs.__getitem__, reverse=True)[:10]
    ranked = _dedupe_by_title_overlap(ranked, papers)[:8]
    if ranked:
        for pid in ranked:
            p = papers.get(pid, {})
            title = p.get("title", pid)
            date = p.get("date", "")
            year = date[:4] if date else "?"
            try:
                cats_list = json.loads(p.get("categories", "[]") or "[]")
                field = cats_list[0] if cats_list else "?"
            except json.JSONDecodeError:
                field = "?"
            abstract = p.get("abstract", "")
            first_sent = ""
            if abstract:
                first_sent = abstract.split(". ")[0].strip()
                if len(first_sent) > 600:
                    first_sent = first_sent[:600] + "…"
            lines.append(f"- **{title}** (arxiv:{pid}, {year}, {field})")
            if first_sent:
                lines.append(f"  {first_sent}.")
            lines.append("")
    else:
        lines += ["No papers with relevance scores found.", ""]

    # Footer: provenance (moved to end)
    if directions and (n_grounded or n_inferred or n_speculative):
        provenance = (
            f"*Provenance: {n_grounded} claim(s) grounded in cited abstracts "
            f"| {n_inferred} inferred | {n_speculative} speculative*"
        )
        lines += ["---", "", provenance, ""]

    (out_dir / "bridge_report.md").write_text("\n".join(lines), encoding="utf-8")
    return directions


# ── Deduplication ─────────────────────────────────────────────────────────────

def _dedupe_by_title_overlap(paper_ids: list[str], papers: dict) -> list[str]:
    """Remove near-duplicate titles (>80% token overlap); keep more recent paper."""
    def _overlap(t1: str, t2: str) -> float:
        s1 = set(t1.lower().split())
        s2 = set(t2.lower().split())
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / max(len(s1), len(s2))

    kept: list[str] = []
    for pid in paper_ids:
        p = papers.get(pid, {})
        title = p.get("title", "") or ""
        date = p.get("date", "") or ""
        duplicate = False
        for ki, kept_pid in enumerate(kept):
            kp = papers.get(kept_pid, {})
            if _overlap(title, kp.get("title", "") or "") > 0.80:
                kept_date = kp.get("date", "") or ""
                if date and kept_date and date > kept_date:
                    kept[ki] = pid
                duplicate = True
                break
        if not duplicate:
            kept.append(pid)
    return kept


# ── sources.bib ───────────────────────────────────────────────────────────────

def _write_sources_bib(
    out_dir: Path,
    top_clusters: list[BridgeCluster],
    papers: dict,
) -> None:
    seen: set[str] = set()
    entries: list[str] = []

    paper_ids = [pid for c in top_clusters for pid in c.paper_ids]
    for arxiv_id in paper_ids:
        if arxiv_id in seen:
            continue
        seen.add(arxiv_id)
        p = papers.get(arxiv_id, {})
        if not p:
            continue

        title = p.get("title", "").replace("{", "\\{").replace("}", "\\}").replace("&", "\\&")
        date = p.get("date", "")
        year = date[:4] if date else "?"

        authors_raw = p.get("authors", "[]")
        try:
            author_list: list[str] = json.loads(authors_raw)
        except Exception:
            author_list = []
        author_str = " and ".join(author_list) if author_list else "Unknown"

        key = arxiv_id.replace("/", "_").replace(".", "_")
        entry = (
            f"@misc{{{key},\n"
            f"  author       = {{{author_str}}},\n"
            f"  title        = {{{{{title}}}}},\n"
            f"  year         = {{{year}}},\n"
            f"  howpublished = {{\\url{{https://arxiv.org/abs/{arxiv_id}}}}},\n"
            f"  note         = {{arXiv:{arxiv_id}}}\n"
            f"}}"
        )
        entries.append(entry)

    (out_dir / "sources.bib").write_text("\n\n".join(entries) + "\n", encoding="utf-8")


# ── bridge_map.html ───────────────────────────────────────────────────────────

_BRIDGE_MAP_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>arxanon — bridge map</title>
<style>
body { margin: 0; background: #0f1117; color: #ccc; font-family: monospace; overflow: hidden; }
svg { width: 100vw; height: 100vh; }
.link { stroke-opacity: 0.5; }
#tooltip {
  position: absolute; background: #1e2030; border: 1px solid #444;
  padding: 8px 12px; border-radius: 4px; pointer-events: none;
  font-size: 12px; max-width: 320px; line-height: 1.5; display: none;
}
#legend {
  position: absolute; top: 16px; right: 16px; background: rgba(30,32,48,0.92);
  border: 1px solid #444; padding: 12px 16px; border-radius: 6px; font-size: 11px;
}
#legend h4 { margin: 0 0 6px 0; font-size: 12px; color: #eee; }
#legend h4:not(:first-child) { margin-top: 10px; }
.lr { display: flex; align-items: center; margin: 3px 0; }
.ld { width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; flex-shrink: 0; }
.ll { width: 24px; height: 3px; margin-right: 8px; flex-shrink: 0; }
#info { position: absolute; bottom: 16px; left: 16px; font-size: 11px; color: #666; }
</style>
</head>
<body>
<svg id="graph"></svg>
<div id="tooltip"></div>
<div id="legend">
  <h4>Categories</h4>
  <div class="lr"><div class="ld" style="background:#4e88d9"></div>cs</div>
  <div class="lr"><div class="ld" style="background:#e8973a"></div>math</div>
  <div class="lr"><div class="ld" style="background:#50b86c"></div>physics / q-bio</div>
  <div class="lr"><div class="ld" style="background:#9b72d4"></div>stat</div>
  <div class="lr"><div class="ld" style="background:#888"></div>other</div>
  <h4>Bridge Type</h4>
  <div class="lr"><div class="ll" style="background:#4caf50"></div>STRUCTURAL</div>
  <div class="lr"><div class="ll" style="background:#ffb300"></div>METHODOLOGICAL</div>
  <div class="lr"><div class="ll" style="background:#5c9df5"></div>THEMATIC</div>
  <div class="lr"><div class="ll" style="background:#444"></div>unvalidated</div>
</div>
<div id="info">Scroll to zoom · drag nodes · hover for details</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script>
const graph = GRAPH_DATA;
const CAT_COLORS = {cs:"#4e88d9",math:"#e8973a",physics:"#50b86c","q-bio":"#50b86c",stat:"#9b72d4"};
const CLS_COLORS = {STRUCTURAL:"#4caf50",METHODOLOGICAL:"#ffb300",THEMATIC:"#5c9df5"};
const nodeColor = d => CAT_COLORS[d.category] || "#888";
const linkColor = d => CLS_COLORS[d.classification] || "#444";

const svg = d3.select("#graph");
const W = window.innerWidth, H = window.innerHeight;
svg.attr("viewBox", `0 0 ${W} ${H}`);
const g = svg.append("g");
svg.call(d3.zoom().scaleExtent([0.15, 5]).on("zoom", e => g.attr("transform", e.transform)));

const sim = d3.forceSimulation(graph.nodes)
  .force("link", d3.forceLink(graph.links).id(d => d.id).distance(90).strength(0.25))
  .force("charge", d3.forceManyBody().strength(-220))
  .force("center", d3.forceCenter(W / 2, H / 2))
  .force("collision", d3.forceCollide(18));

const link = g.append("g").selectAll("line")
  .data(graph.links).join("line")
  .attr("class", "link")
  .style("stroke", linkColor)
  .style("stroke-width", d => 1 + d.similarity * 2.5);

const node = g.append("g").selectAll("g")
  .data(graph.nodes).join("g")
  .call(d3.drag()
    .on("start", (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
    .on("drag",  (e, d) => { d.fx=e.x; d.fy=e.y; })
    .on("end",   (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }));

node.append("circle")
  .attr("r", d => 5 + (d.bridge_count||0)*2)
  .style("fill", nodeColor)
  .style("stroke", "#1e2030")
  .style("stroke-width", 1.5);

const tip = d3.select("#tooltip");
node.on("mouseover", (e, d) => {
  tip.style("display","block").style("left",(e.pageX+14)+"px").style("top",(e.pageY-10)+"px")
     .html(`<strong style="color:#eee">${d.title}</strong><br>`
         + `<span style="color:#888">${d.id}</span><br>`
         + `<span style="color:${nodeColor(d)}">${d.category}</span>`
         + (d.classification ? ` · <span style="color:${CLS_COLORS[d.classification]||'#888'}">${d.classification}</span>` : ""));
}).on("mousemove", e => {
  tip.style("left",(e.pageX+14)+"px").style("top",(e.pageY-10)+"px");
}).on("mouseout", () => tip.style("display","none"));

sim.on("tick", () => {
  link.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
      .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
  node.attr("transform",d=>`translate(${d.x},${d.y})`);
});
</script>
</body>
</html>"""


def _write_bridge_map(
    out_dir: Path,
    top_clusters: list[BridgeCluster],
    papers: dict,
    direct_pairs: list | None = None,
) -> None:
    validated_pairs: dict[tuple, str] = {}
    for cluster in top_clusters:
        for pair in cluster.validated_pairs:
            key = (pair.paper_a, pair.paper_b)
            validated_pairs[key] = pair.classification
            validated_pairs[(pair.paper_b, pair.paper_a)] = pair.classification

    bridge_counts: dict[str, int] = {}
    for cluster in top_clusters:
        for id_a, id_b, _ in cluster.bridge_edges:
            bridge_counts[id_a] = bridge_counts.get(id_a, 0) + 1
            bridge_counts[id_b] = bridge_counts.get(id_b, 0) + 1

    paper_ids = list({pid for c in top_clusters for pid in c.paper_ids})
    nodes = []
    for pid in paper_ids:
        p = papers.get(pid, {})
        cats_raw = p.get("categories", "[]")
        try:
            cats = json.loads(cats_raw)
            primary = cats[0].split(".")[0] if cats else "?"
        except Exception:
            primary = "?"
        nodes.append({
            "id": pid,
            "title": (p.get("title") or pid)[:80],
            "category": primary,
            "bridge_count": min(bridge_counts.get(pid, 0), 5),
        })

    links = []
    seen_edges: set[tuple] = set()
    for cluster in top_clusters:
        for id_a, id_b, sim in cluster.bridge_edges:
            edge_key = (min(id_a, id_b), max(id_a, id_b))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            cls = validated_pairs.get((id_a, id_b), "")
            links.append({
                "source": id_a,
                "target": id_b,
                "similarity": round(float(sim), 4),
                "classification": cls,
            })

    if direct_pairs:
        existing_node_ids = {n["id"] for n in nodes}
        for pair in direct_pairs:
            for pid in (pair.paper_a, pair.paper_b):
                if pid not in existing_node_ids:
                    p = papers.get(pid, {})
                    try:
                        cats = json.loads(p.get("categories", "[]"))
                        primary = cats[0].split(".")[0] if cats else "?"
                    except Exception:
                        primary = "?"
                    nodes.append({
                        "id": pid,
                        "title": (p.get("title") or pid)[:80],
                        "category": primary,
                        "bridge_count": 1,
                    })
                    existing_node_ids.add(pid)
            edge_key = (min(pair.paper_a, pair.paper_b), max(pair.paper_a, pair.paper_b))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                links.append({
                    "source": pair.paper_a,
                    "target": pair.paper_b,
                    "similarity": round(pair.similarity, 4),
                    "classification": pair.classification,
                })

    graph_data = {"nodes": nodes, "links": links}
    html = _BRIDGE_MAP_TEMPLATE.replace("GRAPH_DATA", json.dumps(graph_data))
    (out_dir / "bridge_map.html").write_text(html, encoding="utf-8")
