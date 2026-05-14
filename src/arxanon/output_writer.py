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

# ── Public entry point ────────────────────────────────────────────────────────

def _query_slug(query: str) -> str:
    words = re.sub(r"[^a-z0-9 ]", "", query.lower()).split()[:4]
    return "_".join(words) if words else "arxanon_session"


def save_session(
    query: str,
    bridge_result: BridgePipelineResult,
    papers: dict[str, dict],
) -> Path:
    """Save all output files to ./{query_slug}/. Returns the directory path."""
    slug = _query_slug(query)
    out_dir = Path(slug)
    if out_dir.exists():
        i = 2
        while Path(f"{slug}_{i}").exists():
            i += 1
        out_dir = Path(f"{slug}_{i}")
    out_dir.mkdir(exist_ok=True)

    top_clusters = bridge_result.clusters[:5]

    _write_bridge_report(out_dir, query, bridge_result, papers, top_clusters)
    _write_sources_bib(out_dir, top_clusters, papers)
    _write_bridge_map(out_dir, top_clusters, papers, bridge_result.direct_cross_domain_pairs or [])

    return out_dir


# ── Gemma research directions ─────────────────────────────────────────────────

def _gemma_research_directions(
    query: str,
    top_papers_text: str,
    bridge_pairs_text: str,
    direct_pairs: list | None = None,
    papers: dict | None = None,
    str_papers_text: str = "",
) -> Optional[str]:
    try:
        from .llm_client import call_llm

        # Build enriched bridge context from structural direct pairs (abstracts + reasoning)
        structural = [p for p in (direct_pairs or []) if p.classification == "STRUCTURAL"][:3]
        bridge_rich_lines: list[str] = []
        for i, pair in enumerate(structural, 1):
            pa = (papers or {}).get(pair.paper_a, {})
            pb = (papers or {}).get(pair.paper_b, {})
            try:
                cats_b = json.loads(pb.get("categories", "[]") or "[]")
                cat_b = cats_b[0] if cats_b else "?"
            except Exception:
                cat_b = "?"
            bridge_rich_lines += [
                f"BRIDGE {i}:",
                f"  Math/physics paper: \"{pb.get('title', pair.paper_b)}\" (arxiv:{pair.paper_b}, {cat_b})",
                f"  Abstract: {(pb.get('abstract') or '')[:300]}",
                f"  ML paper: \"{pa.get('title', pair.paper_a)}\" (arxiv:{pair.paper_a})",
                f"  Abstract: {(pa.get('abstract') or '')[:300]}",
                f"  Structural correspondence: {pair.reasoning_chain[:300]}",
                "",
            ]
        bridge_context = "\n".join(bridge_rich_lines) if bridge_rich_lines else bridge_pairs_text

        if bridge_rich_lines:
            prompt = (
                f'You are a research assistant. A researcher asked: "{query}"\n\n'
                f"Cross-domain structural connections found:\n{bridge_context}\n"
                f"Background ML context:\n{top_papers_text}\n\n"
                "TASK: Answer the researcher's question directly.\n"
                "- Name the phenomenon as it is called in the other field.\n"
                "- Cite every claim with an arXiv ID from above.\n"
                "- Propose 1 concrete experiment testing the cross-domain connection.\n\n"
                "Tag every factual claim inline with one of:\n"
                "  [GROUNDED: arxiv:XXXX] — directly stated or strongly implied by the abstract "
                "provided above. Use the exact arXiv ID from the BRIDGE block above.\n"
                "  [INFERRED] — logical inference from what the papers say, not directly stated. "
                "The researcher should verify it.\n"
                "  [SPECULATIVE] — goes beyond the papers. Might be true but the researcher must "
                "evaluate it independently.\n\n"
                "Example of correct tagging:\n"
                "\"Critical slowing down is characterized by increasing recovery time from perturbations "
                "[GROUNDED: arxiv:2208.03881]. This mechanism likely applies to the plateau phase in "
                "grokking training [INFERRED]. The universal exponents suggest a second-order phase "
                "transition class [SPECULATIVE].\"\n\n"
                "MANDATORY: Every sentence containing a factual claim must end with a provenance "
                "tag before its period. Sentences without a tag will be rejected. The three tags:\n"
                "  [GROUNDED: arxiv:XXXX] — directly stated or strongly implied by the abstract above\n"
                "  [INFERRED] — logical deduction from the papers; researcher must verify\n"
                "  [SPECULATIVE] — goes beyond the papers\n\n"
                "Hard constraint: Never use [GROUNDED: arxiv:X] for a claim not directly stated or "
                "strongly implied by the abstract text provided above. When in doubt, use [INFERRED] "
                "rather than [GROUNDED].\n\n"
                "Format:\n"
                "**Cross-Domain Finding:** Start with: \"Outside ML, this is studied as [phenomenon name] "
                "in [field name] (arxiv:[non-CS ID]).\" Then explain the mathematical correspondence "
                "in 1-2 sentences. Do NOT mention ML paper titles or ML-field citations in this first sentence.\n"
                "**What the other field says:** [2-3 sentences from the math/physics paper's perspective, "
                "citing only non-CS papers]\n"
                "**Experiment:** [one specific testable prediction]\n\n"
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
                f"Top papers by relevance:\n{top_papers_text}\n"
                f"{str_section}\n"
                f"Validated bridge connections:\n{bridge_pairs_text}\n\n"
                "Your response MUST begin with 'Outside ML,' or 'Outside of ML,'. "
                "Based only on what is listed above, state: what is this phenomenon called "
                "outside ML, and which paper (arxiv ID) best explains it. Then propose 1 "
                "specific experiment.\n\n"
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
            f"LLM validation reasoning: {(best.reasoning_chain or '[none]')[:400]}",
            "",
            "To generate full synthesis, ensure OpenRouter API is accessible.",
        ]
    else:
        out.append("No validated pairs available. Re-run with more papers or a different query.")
    return out


# ── bridge_report.md ──────────────────────────────────────────────────────────

def _write_bridge_report(
    out_dir: Path,
    query: str,
    result: BridgePipelineResult,
    papers: dict,
    top_clusters: list[BridgeCluster],
) -> None:
    lines: list[str] = [f"# Research Analysis: {query}", ""]

    # ── Section 1: Top 8 papers by relevance score ────────────────────────────
    lines += ["## Top Papers by Relevance", ""]

    qrs = result.query_relevance_scores or {}
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
            score = qrs.get(pid, 0.0)
            abstract = p.get("abstract", "")
            first_sent = ""
            if abstract:
                first_sent = abstract.split(". ")[0].strip()[:120]
                if len(abstract.split(". ")[0]) > 120:
                    first_sent += "…"

            lines.append(f"- **{title}** (arxiv:{pid}, {year}, {field})")
            desc = f"Relevance: {score:.3f}."
            if first_sent:
                desc += f" {first_sent}."
            lines.append(f"  {desc}")
            lines.append("")
    else:
        lines += ["No papers with relevance scores found.", ""]

    # ── Section 2: Validated bridge pairs ─────────────────────────────────────
    lines += ["## Validated Bridge Connections", ""]

    all_validated = [p for c in top_clusters for p in c.validated_pairs]
    direct_pairs = result.direct_cross_domain_pairs or []
    structural_direct = sorted(
        (p for p in direct_pairs if p.classification == "STRUCTURAL"),
        key=lambda p: p.similarity,
        reverse=True,
    )[:5]
    has_structural = bool(structural_direct)

    if not has_structural:
        if all_validated:
            lines.append("### Embedding-Based Validation\n")
            for n, pair in enumerate(all_validated, start=1):
                pa = papers.get(pair.paper_a, {})
                pb = papers.get(pair.paper_b, {})
                title_a = pa.get("title", pair.paper_a)
                title_b = pb.get("title", pair.paper_b)
                try:
                    cat_a = json.loads(pa.get("categories", "[]") or "[]")
                    cat_a_label = cat_a[0] if cat_a else "?"
                except Exception:
                    cat_a_label = "?"
                try:
                    cat_b = json.loads(pb.get("categories", "[]") or "[]")
                    cat_b_label = cat_b[0] if cat_b else "?"
                except Exception:
                    cat_b_label = "?"
                lines += [
                    f"### Pair {n}: {pair.classification} · {pair.confidence_tier}",
                    f"- **{title_a}** (arxiv:{pair.paper_a}, {cat_a_label})",
                    f"- **{title_b}** (arxiv:{pair.paper_b}, {cat_b_label})",
                ]
                if pair.reasoning_chain and pair.reasoning_chain != "[No reasoning captured]":
                    reasoning = pair.reasoning_chain[:400].replace("\n", " ")
                    lines.append(f"- Reasoning: {reasoning}")
                if pair.matched_properties:
                    lines.append(f"- Shared properties: {', '.join(pair.matched_properties[:4])}")
                lines.append("")
        else:
            if direct_pairs:
                lines += [
                    f"*Embedding similarity threshold: no cross-domain bridges above 0.72. "
                    f"Direct LLM comparison found {len(direct_pairs)} connection(s) — shown below.*",
                    "",
                ]
            else:
                lines += [
                    "No embedding-based cross-domain bridges were found for this query.",
                    "",
                    "Possible reasons:",
                    "- All retrieved papers share the same top-level arXiv category.",
                    "- Domain expansion via LLM may not have found papers from adjacent fields.",
                    "- The similarity threshold (0.72) may be too strict for this topic.",
                    "",
                    f'Suggested next step: try `arxanon search "{query}" --max-results 200`,',
                    "or rephrase to include terms from a different field.",
                    "",
                ]

    display_direct = structural_direct if has_structural else direct_pairs
    if display_direct:
        lines += [
            "### Direct LLM Comparison (embedding threshold bypassed)",
            "",
            "*The following pairs were validated by sending abstracts directly to the LLM,*",
            "*without requiring embedding similarity. They represent connections where the*",
            "*mathematical structure is shared but domain vocabulary diverges too far for*",
            "*embedding-based similarity to detect.*",
            "",
        ]
        for n, pair in enumerate(display_direct, start=1):
            pa = papers.get(pair.paper_a, {})
            pb = papers.get(pair.paper_b, {})
            title_a = pa.get("title", pair.paper_a)
            title_b = pb.get("title", pair.paper_b)
            try:
                cat_a = json.loads(pa.get("categories", "[]") or "[]")
                cat_a_label = cat_a[0] if cat_a else "?"
            except Exception:
                cat_a_label = "?"
            try:
                cat_b = json.loads(pb.get("categories", "[]") or "[]")
                cat_b_label = cat_b[0] if cat_b else "?"
            except Exception:
                cat_b_label = "?"
            tag_a = pa.get("query_tag", "")
            tag_b = pb.get("query_tag", "")
            lines += [
                f"### Direct Pair {n}: {pair.classification}",
                f"- **{title_a}** (arxiv:{pair.paper_a}, {cat_a_label}, channel:{tag_a})",
                f"- **{title_b}** (arxiv:{pair.paper_b}, {cat_b_label}, channel:{tag_b})",
            ]
            if pair.reasoning_chain:
                reasoning = pair.reasoning_chain[:400].replace("\n", " ")
                lines.append(f"- Reasoning: {reasoning}")
            lines.append(f"- Validation: direct LLM comparison (relevance sum: {pair.similarity * 2:.3f})")
            lines.append("")

    # ── Section 3: Gemma research directions ──────────────────────────────────
    lines += ["## Research Directions", ""]

    # Build context for Gemma
    top5 = sorted(qrs, key=qrs.__getitem__, reverse=True)[:5]
    top_papers_lines: list[str] = []
    for pid in top5:
        p = papers.get(pid, {})
        title = p.get("title", pid)
        abstract = (p.get("abstract") or "")[:200]
        top_papers_lines.append(f"- {title} (arxiv:{pid}): {abstract}")
    top_papers_text = "\n".join(top_papers_lines) if top_papers_lines else "No papers."

    str_pids = [pid for pid, p in papers.items()
                if (p.get("query_tag") or "").startswith("str")]
    str_by_qrs = sorted(str_pids, key=lambda p: qrs.get(p, 0.0), reverse=True)[:5]
    str_papers_lines: list[str] = []
    for pid in str_by_qrs:
        p = papers.get(pid, {})
        title = p.get("title", pid)
        abstract = (p.get("abstract") or "")[:200]
        try:
            cats = json.loads(p.get("categories", "[]") or "[]")
            cat = cats[0] if cats else "?"
        except Exception:
            cat = "?"
        str_papers_lines.append(f"- {title} (arxiv:{pid}, {cat}): {abstract}")
    str_papers_text = "\n".join(str_papers_lines)

    combined_pairs = structural_direct if has_structural else all_validated
    if combined_pairs:
        bridge_lines: list[str] = []
        for pair in combined_pairs[:5]:
            pa = papers.get(pair.paper_a, {})
            pb = papers.get(pair.paper_b, {})
            ta = pa.get("title", pair.paper_a)
            tb = pb.get("title", pair.paper_b)
            bridge_lines.append(
                f"- {pair.classification}: arxiv:{pair.paper_a} ({ta[:60]}) "
                f"↔ arxiv:{pair.paper_b} ({tb[:60]})"
            )
        bridge_pairs_text = "\n".join(bridge_lines)
    else:
        bridge_pairs_text = "None found."

    directions = _gemma_research_directions(
        query, top_papers_text, bridge_pairs_text,
        direct_pairs=direct_pairs, papers=papers,
        str_papers_text=str_papers_text,
    )
    if directions:
        n_grounded = len(re.findall(r'\[GROUNDED:', directions))
        n_inferred = len(re.findall(r'\[INFERRED\]', directions))
        n_speculative = len(re.findall(r'\[SPECULATIVE\]', directions))
        provenance = (
            f"*Provenance: {n_grounded} claim(s) grounded in cited abstracts "
            f"| {n_inferred} inferred | {n_speculative} speculative*"
        )
        lines += [provenance, ""]
        lines.append(directions)
    else:
        lines += _deterministic_directions_fallback(query, combined_pairs, papers)
    lines.append("")

    (out_dir / "bridge_report.md").write_text("\n".join(lines), encoding="utf-8")


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
