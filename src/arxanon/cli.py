"""arxanon CLI — Rich terminal interface for cross-domain analogy discovery."""
from __future__ import annotations

import json
import sys
from typing import Optional

# Ensure Unicode characters (✓, —, etc.) render correctly on Windows cp1252 terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from . import __version__, config
from .arxiv_client import fetch_and_store_papers
from .db import get_citation_edge_count, get_unique_category_count, init_db
from .pipeline import (
    _gemma_expand_queries,
    _llm_structural_queries,
    embed_and_index_papers,
    load_papers_with_embeddings,
)
from .semantic_scholar import fetch_and_store_citations
from .clusters import BridgePipelineResult

console = Console(legacy_windows=False)


def _header() -> None:
    model_label = config.EMBED_MODEL.split("/")[-1]
    header = Text.assemble(
        ("A R X A N O N", "bold white"),
        (f"  v{__version__}", "dim white"),
        "\n",
        ("Cross-Domain Structural Analogy Engine", "dim cyan"),
    )
    console.print(
        Panel(
            header,
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print(f"  Embedding: [cyan]{model_label}[/cyan]\n")


def _retrieval_panel(
    total_fetched: int,
    n_queries: int,
    n_cats: int,
    sem_fetched: int = 0,
    str_fetched: int = 0,
) -> None:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim")
    grid.add_column(justify="right", style="green")
    grid.add_row("Queries:", str(n_queries))
    if sem_fetched > 0 and str_fetched > 0:
        grid.add_row("  semantic channel:", f"{sem_fetched} papers")
        grid.add_row("  structural channel:", f"{str_fetched} papers")
    grid.add_row("Papers fetched:", f"[bold]{total_fetched}[/bold]")
    grid.add_row("Fields:", f"{n_cats} arXiv categories")
    console.print(Panel(grid, title="[bold]Retrieval[/bold]", border_style="blue", padding=(0, 1)))


def _citation_panel(direct: int, cocitation: int) -> None:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim")
    grid.add_column(justify="right", style="green")
    grid.add_row("Direct edges:", str(direct))
    grid.add_row("Co-citation edges:", str(cocitation))
    grid.add_row("Total:", str(direct + cocitation))
    console.print(
        Panel(grid, title="[bold]Citation Graph[/bold]", border_style="blue", padding=(0, 1))
    )


def _embedding_panel(index_size: int, embed_model: str) -> None:
    model_label = embed_model.split("/")[-1]
    dim = config.EMBED_DIMS.get(embed_model, "?")
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim")
    grid.add_column(justify="right", style="green")
    grid.add_row("Index size:", f"{index_size} vectors")
    grid.add_row("Dimensions:", str(dim))
    grid.add_row("Model:", model_label)
    console.print(
        Panel(grid, title="[bold]Embeddings[/bold]", border_style="blue", padding=(0, 1))
    )


# ── Phase 2+3 display helpers ─────────────────────────────────────────────────

def _bridge_detection_panel(result: object) -> None:
    from .clusters import BridgePipelineResult  # local import avoids top-level cycle
    assert isinstance(result, BridgePipelineResult)

    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim")
    grid.add_column(justify="right", style="green")

    grid.add_row(
        "Similarity graph:",
        f"{result.graph_nodes} nodes · {result.graph_edges_full} edges",
    )
    grid.add_row(
        "After citation exclusion:",
        f"{result.graph_nodes} nodes · [bold]{result.graph_edges_bridge}[/bold] bridge edges",
    )

    tda = result.tda_result
    if tda.enabled and tda.n_cycles > 0:
        grid.add_row("Persistent homology:", f"{tda.n_cycles} cycle(s) (H₁) detected")
        for cycle in tda.cycles[:3]:
            cats_str = " ↔ ".join(cycle.categories[:3]) if cycle.categories else "?"
            grid.add_row(
                f"  Cycle {cycle.cycle_id + 1}:",
                f"persistence {cycle.persistence:.2f}  —  {cats_str}",
            )
    elif tda.enabled:
        grid.add_row("Persistent homology:", "0 persistent cycles found")
    else:
        grid.add_row("Persistent homology:", f"[dim]{tda.warning or 'disabled'}[/dim]")

    n_with_tda = sum(1 for c in result.clusters if c.tda_cycle_ids)
    grid.add_row(
        "HDBSCAN clusters:",
        f"[bold]{len(result.clusters)}[/bold] ({n_with_tda} with topological support)",
    )
    if result.bibcoupling_edges_added > 0:
        grid.add_row("Bibcoupling edges added:", str(result.bibcoupling_edges_added))

    console.print(
        Panel(
            grid,
            title="[bold]STAGE 3: Bridge Detection[/bold]",
            border_style="blue",
            padding=(0, 1),
        )
    )


def _gemma_validation_panel(result: object) -> None:
    from .clusters import BridgePipelineResult
    assert isinstance(result, BridgePipelineResult)

    if not result.gemma_available:
        warning = Text()
        warning.append("⚠  Ollama not available — bridges shown unvalidated.\n\n", style="yellow")
        warning.append("Run: ", style="dim")
        warning.append("ollama serve\n     ", style="cyan")
        warning.append("ollama pull ", style="cyan")
        warning.append(config.GEMMA_MODEL, style="bold cyan")
        if result.gemma_warning:
            warning.append(f"\n\n{result.gemma_warning}", style="dim")
        console.print(
            Panel(
                warning,
                title="[bold]STAGE 4: Structural Analogy Verification[/bold]",
                border_style="yellow",
                padding=(0, 1),
            )
        )
        return

    from collections import Counter
    all_pairs = [pair for cluster in result.clusters for pair in cluster.validated_pairs]
    counts: Counter = Counter(pair.classification for pair in all_pairs)

    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim", min_width=18)
    grid.add_column()
    grid.add_row("[green]STRUCTURAL[/green]", f"[bold green]{counts.get('STRUCTURAL', 0)}[/bold green]  deep mathematical correspondence")
    grid.add_row("[yellow]METHODOLOGICAL[/yellow]", f"[bold yellow]{counts.get('METHODOLOGICAL', 0)}[/bold yellow]  shared technique")
    grid.add_row("[blue]THEMATIC[/blue]", f"[bold blue]{counts.get('THEMATIC', 0)}[/bold blue]  related research direction")
    total_candidates = sum(len(c.bridge_edges) for c in result.clusters)
    discarded = max(0, total_candidates - len(all_pairs))
    grid.add_row("[dim]SUPERFICIAL[/dim]", f"[dim]{discarded}[/dim]  discarded")

    think_pct = (
        int(100 * sum(1 for p in all_pairs if p.think_mode_used) / max(len(all_pairs), 1))
    )
    console.print(
        Panel(
            grid,
            title=f"[bold]STAGE 4: Structural Analogy Verification ({len(all_pairs)} pairs · {think_pct}% with extended reasoning)[/bold]",
            border_style="green",
            padding=(0, 1),
        )
    )


def _direct_pairs_panel(pairs: list, papers: dict) -> None:
    """Display direct LLM-validated cross-domain pairs."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim", min_width=18)
    grid.add_column()

    for pair in pairs:
        pa = papers.get(pair.paper_a, {})
        pb = papers.get(pair.paper_b, {})
        title_a = (pa.get("title") or pair.paper_a)[:60]
        title_b = (pb.get("title") or pair.paper_b)[:60]
        try:
            cat_a = json.loads(pa.get("categories", "[]") or "[]")[0]
        except Exception:
            cat_a = "?"
        try:
            cat_b = json.loads(pb.get("categories", "[]") or "[]")[0]
        except Exception:
            cat_b = "?"
        tag_a = pa.get("query_tag", "")
        tag_b = pb.get("query_tag", "")
        cls_color = {"STRUCTURAL": "green", "METHODOLOGICAL": "yellow", "THEMATIC": "blue"}.get(
            pair.classification, "white"
        )
        grid.add_row(
            f"[{cls_color}]{pair.classification}[/{cls_color}]",
            f"[cyan]{title_a}[/cyan] [dim]({cat_a}, {tag_a})[/dim]\n"
            f"  ↔ [yellow]{title_b}[/yellow] [dim]({cat_b}, {tag_b})[/dim]\n"
            f"  [italic dim]{pair.reasoning_chain[:120]}[/italic dim]",
        )

    console.print(
        Panel(
            grid,
            title=f"[bold]Direct Cross-Domain Validation ({len(pairs)} pairs — embedding threshold bypassed)[/bold]",
            border_style="green",
            padding=(0, 1),
        )
    )


def _bridge_clusters_table(clusters: list) -> None:
    if not clusters:
        return

    table = Table(
        title="Bridge Clusters — ranked by discovery potential",
        border_style="cyan",
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Score", width=6, justify="right")
    table.add_column("Domains", min_width=14)
    table.add_column("Papers", width=7, justify="right")
    table.add_column("Isolated", width=9, justify="right")
    table.add_column("TDA", width=5, justify="center")
    table.add_column("Classification", min_width=18)

    for rank, cluster in enumerate(clusters, start=1):
        domains_str = " ↔ ".join(cluster.categories[:4])
        if len(cluster.categories) > 4:
            domains_str += f" +{len(cluster.categories) - 4}"

        isolation_pct = f"{cluster.score.citation_isolation * 100:.0f}%"
        tda_str = "[green]✓[/green]" if cluster.tda_cycle_ids else "[dim]—[/dim]"

        if cluster.validated_pairs:
            top = cluster.validated_pairs[0]
            cls_color = {
                "STRUCTURAL": "green",
                "METHODOLOGICAL": "yellow",
                "THEMATIC": "blue",
            }.get(top.classification, "dim")
            cls_display = (
                f"[{cls_color}]{top.classification}[/{cls_color}] · {top.confidence_tier}"
            )
        else:
            cls_display = "[dim]unvalidated[/dim]"

        table.add_row(
            str(rank),
            f"{cluster.score.composite:.3f}",
            domains_str,
            str(len(cluster.paper_ids)),
            isolation_pct,
            tda_str,
            cls_display,
        )

    console.print(table)


def _bridge_cluster_detail(cluster: object, papers: dict) -> None:
    if cluster is None:
        return

    from .clusters import BridgeCluster
    assert isinstance(cluster, BridgeCluster)

    tier = cluster.validated_pairs[0].confidence_tier if cluster.validated_pairs else "UNVALIDATED"
    domains_str = " ↔ ".join(cluster.categories)
    n_papers = len(cluster.paper_ids)
    n_cats = len(cluster.categories)

    console.print(Rule(
        f"[bold]BRIDGE #1  ·  Score: {cluster.score.composite:.3f}  ·  Confidence: {tier}[/bold]",
        style="cyan",
    ))
    console.print(
        f"  Domains: [cyan]{domains_str}[/cyan]  "
        f"([bold]{n_papers}[/bold] papers, {n_cats} categories)\n"
    )

    if cluster.validated_pairs:
        for i, pair in enumerate(cluster.validated_pairs[:3], start=1):
            pa = papers.get(pair.paper_a, {})
            pb = papers.get(pair.paper_b, {})
            title_a = (pa.get("title") or pair.paper_a)[:70]
            title_b = (pb.get("title") or pair.paper_b)[:70]
            try:
                cats_a = json.loads(pa.get("categories", "[]") or "[]")
                cat_a_label = cats_a[0] if cats_a else "?"
            except Exception:
                cat_a_label = "?"
            try:
                cats_b = json.loads(pb.get("categories", "[]") or "[]")
                cat_b_label = cats_b[0] if cats_b else "?"
            except Exception:
                cat_b_label = "?"
            tag_a = pa.get("query_tag", "")
            tag_b = pb.get("query_tag", "")

            cls_color = {
                "STRUCTURAL": "green",
                "METHODOLOGICAL": "yellow",
                "THEMATIC": "blue",
            }.get(pair.classification, "white")

            console.print(f"  [bold]{i}.[/bold] [cyan]{title_a}[/cyan]")
            console.print(f"     [dim]arxiv:{pair.paper_a}  ·  {cat_a_label}  ·  channel:{tag_a}[/dim]")
            console.print(f"     ↔ [yellow]{title_b}[/yellow]")
            console.print(f"     [dim]arxiv:{pair.paper_b}  ·  {cat_b_label}  ·  channel:{tag_b}[/dim]")
            console.print(
                f"     [{cls_color}]{pair.classification}[/{cls_color}]"
                f" · {pair.confidence_tier}"
                f" · sim {pair.similarity:.4f}"
                f"{'  🧠' if pair.think_mode_used else ''}"
            )
            if pair.reasoning_chain:
                preview = pair.reasoning_chain[:200].replace("\n", " ")
                if len(pair.reasoning_chain) > 200:
                    preview += "…"
                console.print(f"     [italic dim]{preview}[/italic dim]")
            console.print()
        console.print(Rule(style="dim"))
    else:
        console.print("  [yellow]Bridge edges (unvalidated — run without --no-gemma to verify):[/yellow]\n")
        for i, (id_a, id_b, score) in enumerate(cluster.bridge_edges[:3], start=1):
            pa = papers.get(id_a, {})
            pb = papers.get(id_b, {})
            title_a = (pa.get("title") or id_a)[:70]
            title_b = (pb.get("title") or id_b)[:70]
            console.print(f"  [bold]{i}.[/bold] {title_a}")
            console.print(f"     ↔ {title_b}")
            console.print(f"     [dim]sim {score:.4f}[/dim]\n")
        console.print(Rule(style="dim"))


def _bridge_explanation_panel(cluster: object, papers: dict) -> None:
    """Show plain-language explanation, translation dict, and next step for the top bridge."""
    from .clusters import BridgeCluster
    assert isinstance(cluster, BridgeCluster)

    if not cluster.validated_pairs:
        # Unvalidated: show top bridge pair titles with a nudge
        if cluster.bridge_edges:
            id_a, id_b, _ = cluster.bridge_edges[0]
            pa = papers.get(id_a, {})
            pb = papers.get(id_b, {})
            cats_a = json.loads(pa.get("categories", "[]") or "[]")
            cats_b = json.loads(pb.get("categories", "[]") or "[]")
            domain_a = cats_a[0] if cats_a else "?"
            domain_b = cats_b[0] if cats_b else "?"
            title_a = (pa.get("title") or id_a)[:70]
            title_b = (pb.get("title") or id_b)[:70]
            console.print(
                Panel(
                    f"[bold]{title_a}[/bold]\n"
                    f"[dim]arxiv:{id_a}  ·  {domain_a}[/dim]\n\n"
                    f"↔  [bold]{title_b}[/bold]\n"
                    f"[dim]arxiv:{id_b}  ·  {domain_b}[/dim]\n\n"
                    f"[yellow]Run without --no-gemma to get the structural explanation "
                    f"and translation dictionary.[/yellow]",
                    title="[bold]Top Bridge — unvalidated[/bold]",
                    border_style="yellow",
                    padding=(0, 2),
                )
            )
        return

    pair = cluster.validated_pairs[0]
    pa = papers.get(pair.paper_a, {})
    pb = papers.get(pair.paper_b, {})
    cats_a = json.loads(pa.get("categories", "[]") or "[]")
    cats_b = json.loads(pb.get("categories", "[]") or "[]")
    domain_a = cats_a[0] if cats_a else "?"
    domain_b = cats_b[0] if cats_b else "?"

    cls_color = {"STRUCTURAL": "green", "METHODOLOGICAL": "yellow", "THEMATIC": "blue"}.get(
        pair.classification, "white"
    )
    parts: list[str] = []

    # Full reasoning chain
    if pair.reasoning_chain and pair.reasoning_chain != "[No reasoning captured]":
        parts += [
            "[bold]The Structural Connection[/bold]",
            "",
            pair.reasoning_chain.strip(),
            "",
        ]

    # Translation dictionary
    if pair.translation_hints:
        parts += [f"[bold]Translation: {domain_a} → {domain_b}[/bold]", ""]
        for hint in pair.translation_hints[:6]:
            term_a = hint.get("term_a", "")
            term_b = hint.get("term_b", "")
            if term_a and term_b:
                parts.append(f"  [cyan]{term_a}[/cyan]  →  [yellow]{term_b}[/yellow]")
        parts.append("")
    elif pair.matched_properties:
        parts += [
            f"[bold]Shared properties ({domain_a} ↔ {domain_b})[/bold]",
            "",
            "  " + "  ·  ".join(pair.matched_properties[:5]),
            "",
        ]

    # Concrete next step
    title_b = (pb.get("title") or pair.paper_b)[:72]
    if pair.matched_properties:
        prop = pair.matched_properties[0]
        next_step = (
            f"Read [cyan]arxiv:{pair.paper_b}[/cyan] to see how [italic]{prop}[/italic] "
            f"is formalized in {domain_b}.\n"
            f"[dim]{title_b}[/dim]"
        )
    else:
        next_step = (
            f"Read [cyan]arxiv:{pair.paper_b}[/cyan] ({domain_b}).\n"
            f"[dim]{title_b}[/dim]"
        )

    parts += ["[bold]Next step[/bold]", "", next_step]

    console.print(
        Panel(
            "\n".join(parts),
            title=(
                f"[bold]Bridge: {domain_a} ↔ {domain_b}"
                f"  ·  [{cls_color}]{pair.classification}[/{cls_color}]"
                f"  ·  {pair.confidence_tier}[/bold]"
            ),
            border_style=cls_color,
            padding=(0, 2),
        )
    )


# ── CLI commands ──────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.pass_context
@click.version_option(__version__, prog_name="arxanon")
def main(ctx: click.Context) -> None:
    """Arxanon — cross-domain structural analogy discovery for AI/ML researchers.

    Finds papers from fields you don't follow that describe the exact mathematical
    structure you're stuck on, and tells you precisely why they matter.

    Run with no arguments to start an interactive session.
    """
    if ctx.invoked_subcommand is None:
        from .interactive import run_interactive_session
        run_interactive_session()


# ── Core pipeline (shared by search command and interactive session) ──────────

def execute_pipeline(
    query: str,
    max_results: int,
    top_clusters: int,
    max_validate: Optional[int],
    no_gemma: bool,
    no_tda: bool,
    coupling_threshold: int,
    verbose: bool = True,
) -> Optional[tuple[BridgePipelineResult, dict]]:
    """Run the full pipeline (phases 1–3) with Rich display.

    Returns (bridge_result, papers) on success, None if no vectors were indexed.
    Assumes init_db() has already been called by the caller.
    Uses the module-level console for all output.
    When verbose=False, suppresses all info panels; progress bars remain.
    """
    # ── Phase 1, Stage 1: Fetch papers ────────────────────────────────────────
    if verbose:
        console.print("[bold]Stage 1:[/bold] Retrieving papers from arXiv\n")

    sem_queries = _gemma_expand_queries(query)
    str_queries = _llm_structural_queries(query, sem_queries)
    structural_query_map: dict[str, str] = {
        f"str{i + 1}": q for i, q in enumerate(str_queries)
    }

    # Split budget evenly between channels; structural gets nothing if LLM returned no queries
    if str_queries:
        sem_budget = max_results // 2
        str_budget = max_results - sem_budget
    else:
        sem_budget = max_results
        str_budget = 0

    sem_per = max(1, sem_budget // max(len(sem_queries), 1))
    str_per = max(1, str_budget // max(len(str_queries), 1)) if str_queries else 0

    total_fetched = 0
    sem_fetched = 0
    str_fetched = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_a = progress.add_task("[cyan]Searching arXiv...[/cyan]", total=max_results)

        for i, q in enumerate(sem_queries):
            label = q[:55] + "…" if len(q) > 55 else q
            progress.update(
                task_a,
                description=f"[cyan][semantic {i + 1}/{len(sem_queries)}] {label}[/cyan]",
            )
            offset = total_fetched
            count = fetch_and_store_papers(
                q,
                sem_per,
                f"sem{i + 1}",
                on_paper=lambda n, _off=offset: progress.update(task_a, completed=_off + n),
            )
            sem_fetched += count
            total_fetched += count

        for i, q in enumerate(str_queries):
            label = q[:52] + "…" if len(q) > 52 else q
            progress.update(
                task_a,
                description=f"[magenta][structural {i + 1}/{len(str_queries)}] {label}[/magenta]",
            )
            offset = total_fetched
            count = fetch_and_store_papers(
                q,
                str_per,
                f"str{i + 1}",
                on_paper=lambda n, _off=offset: progress.update(task_a, completed=_off + n),
            )
            str_fetched += count
            total_fetched += count

        sem_str = f"{sem_fetched} semantic"
        str_str = f", {str_fetched} structural" if str_queries else ""
        progress.update(
            task_a,
            completed=total_fetched,
            description=f"[green]✓[/green] {total_fetched} papers fetched ({sem_str}{str_str})",
        )

    n_cats = get_unique_category_count()
    n_queries = len(sem_queries) + len(str_queries)
    if verbose:
        _retrieval_panel(total_fetched, n_queries, n_cats, sem_fetched, str_fetched)

    # ── Phase 1, Stage 2: Citation graph ──────────────────────────────────────
    if verbose:
        console.print("[bold]Stage 2:[/bold] Building citation graph via Semantic Scholar\n")

    from .db import get_all_arxiv_ids

    all_ids = get_all_arxiv_ids()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_c = progress.add_task("[cyan]Fetching citation data...[/cyan]", total=len(all_ids))
        fetch_and_store_citations(
            all_ids,
            on_paper=lambda done, total: progress.update(task_c, completed=done, total=total),
        )
        progress.update(task_c, description="[green]✓[/green] Citation graph complete")

    edge_counts = get_citation_edge_count()
    if verbose:
        _citation_panel(edge_counts.get("direct", 0), edge_counts.get("cocitation", 0))

    # ── Phase 1, Stage 3: Embeddings ──────────────────────────────────────────
    if verbose:
        console.print("[bold]Stage 3:[/bold] Generating embeddings\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task_e = progress.add_task(
            f"[cyan]Encoding with {config.EMBED_MODEL.split('/')[-1]}...[/cyan]"
        )
        index_size, _query_vector = embed_and_index_papers(query)
        progress.update(
            task_e, description=f"[green]✓[/green] {index_size} vectors indexed"
        )

    if verbose:
        _embedding_panel(index_size, config.EMBED_MODEL)

    if index_size == 0:
        return None

    # ── Phase 2: Bridge detection ─────────────────────────────────────────────
    if verbose:
        console.print("[bold]Stage 5:[/bold] Detecting citation-isolated bridge clusters\n")

    from .bridge_pipeline import (
        run_bridge_pipeline,
        run_direct_cross_domain_validation,
        run_gemma_validation,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task_br = progress.add_task("[cyan]Computing bibcoupling edges...[/cyan]")

        def _on_bridge_stage(stage: str, value: object) -> None:
            if stage == "bibcoupling_done":
                progress.update(
                    task_br,
                    description=f"[cyan]Bibcoupling done ({value} new edges). Building graph...[/cyan]",
                )
            elif stage == "graph_done":
                _, e_full, e_bridge = value  # type: ignore[misc]
                progress.update(
                    task_br,
                    description=f"[cyan]Bridge graph: {e_bridge}/{e_full} edges. Running TDA...[/cyan]",
                )
            elif stage == "tda_done":
                tda_r = value
                tda_desc = (
                    f"{tda_r.n_cycles} cycles" if tda_r.enabled else (tda_r.warning or "disabled")  # type: ignore[union-attr]
                )
                progress.update(
                    task_br,
                    description=f"[cyan]TDA: {tda_desc}. Clustering...[/cyan]",
                )
            elif stage == "clustering_done":
                progress.update(
                    task_br,
                    description=f"[cyan]Found {value} bridge clusters.[/cyan]",
                )
            elif stage == "domain_expansion":
                adj_queries = value
                progress.update(
                    task_br,
                    description=(
                        f"[cyan]Single-domain detected — expanding to: "
                        f"{' | '.join(adj_queries)}[/cyan]"
                    ),
                )
            elif stage == "single_domain_detected":
                cats_str = ", ".join(value) if value else "unknown"
                progress.update(
                    task_br,
                    description=f"[yellow]All papers from {cats_str} — expansion found no adjacent fields.[/yellow]",
                )

        bridge_result = run_bridge_pipeline(
            coupling_threshold=coupling_threshold,
            enable_tda=not no_tda,
            on_stage=_on_bridge_stage,
            query=query,
            query_vector=_query_vector,
        )

    papers_embedded = load_papers_with_embeddings()
    if verbose:
        _bridge_detection_panel(bridge_result)

    # ── Phase 3: Gemma validation ─────────────────────────────────────────────
    if not no_gemma:
        if verbose:
            console.print("[bold]Stage 6:[/bold] Gemma 4 structural analogy verification\n")

        _bridge_edges_total = sum(
            len(c.bridge_edges) for c in bridge_result.clusters[:top_clusters]
        )
        total_pairs_estimate = max_validate if max_validate is not None else _bridge_edges_total

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=False,
        ) as progress:
            task_g = progress.add_task(
                "[cyan]Checking Ollama...[/cyan]",
                total=max(total_pairs_estimate, 1),
            )

            def _on_pair(done: int, total: int) -> None:
                progress.update(
                    task_g,
                    completed=done,
                    total=total,
                    description=f"[cyan]Verifying bridge {done}/{total}...[/cyan]",
                )

            bridge_result = run_gemma_validation(
                result=bridge_result,
                papers=papers_embedded,
                top_n_clusters=top_clusters,
                max_validate=max_validate,
                on_pair=_on_pair,
            )
            if bridge_result.gemma_available:
                progress.update(task_g, description="[green]✓[/green] Validation complete")
            else:
                progress.update(task_g, description="[yellow]~[/yellow] Ollama unavailable")

        if verbose:
            _gemma_validation_panel(bridge_result)

    # ── Phase 3b: Direct cross-domain validation (bypasses similarity threshold) ─
    if not no_gemma:
        if verbose:
            console.print(
                "[bold]Stage 6b:[/bold] Direct LLM comparison for cross-domain pairs\n"
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=False,
        ) as progress:
            task_d = progress.add_task("[cyan]Finding cross-channel pairs...[/cyan]", total=50)

            def _on_direct_pair(done: int, total: int) -> None:
                progress.update(
                    task_d,
                    completed=done,
                    total=total,
                    description=f"[cyan]Direct comparison {done}/{total}...[/cyan]",
                )

            bridge_result = run_direct_cross_domain_validation(
                result=bridge_result,
                papers=papers_embedded,
                max_pairs=50,
                on_pair=_on_direct_pair,
                structural_query_map=structural_query_map,
                original_query=query,
            )
            n_direct = len(bridge_result.direct_cross_domain_pairs)
            if n_direct:
                progress.update(
                    task_d,
                    description=f"[green]✓[/green] {n_direct} cross-domain pair(s) validated",
                )
            else:
                progress.update(
                    task_d,
                    description="[dim]No cross-domain pairs found via direct comparison[/dim]",
                )

        if verbose and bridge_result.direct_cross_domain_pairs:
            _direct_pairs_panel(bridge_result.direct_cross_domain_pairs, papers_embedded)

    # ── Display bridge clusters ───────────────────────────────────────────────
    if verbose:
        if bridge_result.clusters:
            _bridge_clusters_table(bridge_result.clusters[:top_clusters])
            _bridge_cluster_detail(bridge_result.clusters[0], papers_embedded)
            _bridge_explanation_panel(bridge_result.clusters[0], papers_embedded)
        else:
            console.print(
                Panel(
                    "[yellow]No bridge clusters found.[/yellow]\n\n"
                    "This may mean the retrieved papers are all from similar domains, "
                    "or all similar pairs are citation-connected.\n\n"
                    "Try [cyan]--max-results 100[/cyan] or a more cross-domain structural query.",
                    title="[bold]Bridge Clusters[/bold]",
                    border_style="yellow",
                )
            )

    return bridge_result, papers_embedded


@main.command()
@click.argument("query")
@click.option(
    "--max-results",
    default=100,
    show_default=True,
    metavar="N",
    help="Maximum papers to fetch per query.",
)
@click.option(
    "--no-gemma",
    is_flag=True,
    default=False,
    help="Skip Gemma 4 structural analogy verification.",
)
@click.option(
    "--no-tda",
    is_flag=True,
    default=False,
    help="Skip persistent homology (TDA) — faster but no topological gap detection.",
)
@click.option(
    "--coupling-threshold",
    default=3,
    show_default=True,
    metavar="N",
    help="Shared-reference count threshold for bibliographic coupling filter.",
)
@click.option(
    "--top-clusters",
    default=5,
    show_default=True,
    metavar="N",
    help="Number of bridge clusters to display and validate with Gemma.",
)
@click.option(
    "--max-validate",
    default=None,
    type=int,
    metavar="N",
    help=(
        "Limit Gemma 4 verification to top N bridge pairs by cosine similarity score. "
        "Use for testing with lightweight models (e.g. gemma4:2b). "
        "Default: no limit."
    ),
)
@click.option(
    "--fresh",
    is_flag=True,
    default=False,
    help="Delete the papers DB and FAISS index before running (start from scratch).",
)
def search(
    query: str,
    max_results: int,
    no_gemma: bool,
    no_tda: bool,
    coupling_threshold: int,
    top_clusters: int,
    max_validate: Optional[int],
    fresh: bool,
) -> None:
    """Search for cross-domain structural analogies to QUERY.

    Runs the full Phase 1 pipeline (retrieval + citation graph + embeddings),
    then Phase 2 (bridge detection via citation exclusion + HDBSCAN), and
    optionally Phase 3 (Gemma 4 structural analogy verification via Ollama).

    Example:

        arxanon search "edge of stability gradient descent neural networks"

        arxanon search "gradient descent oscillation" --no-gemma --no-tda
    """
    _header()
    if fresh:
        if config.DB_PATH.exists():
            config.DB_PATH.unlink()
        if config.FAISS_PATH.exists():
            config.FAISS_PATH.unlink()
    init_db()

    result = execute_pipeline(
        query=query,
        max_results=max_results,
        top_clusters=top_clusters,
        max_validate=max_validate,
        no_gemma=no_gemma,
        no_tda=no_tda,
        coupling_threshold=coupling_threshold,
    )

    if result:
        bridge_result, papers_embedded = result
        if bridge_result.clusters:
            from .output_writer import save_session
            out_dir = save_session(query, bridge_result, papers_embedded)
            console.print(f"\n  [dim]Output saved to[/dim] [cyan]./{out_dir.name}/[/cyan]\n")


if __name__ == "__main__":
    sys.exit(main())
