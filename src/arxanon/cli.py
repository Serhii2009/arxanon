"""arxanon CLI — Rich terminal interface for cross-domain analogy discovery."""
from __future__ import annotations

import sys
from typing import Optional

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
    default_structural_query,
    embed_and_index_papers,
    find_top_similar_pairs,
    load_papers_with_embeddings,
)
from .semantic_scholar import fetch_and_store_citations

console = Console()


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


def _retrieval_panel(count_a: int, count_b: int, n_cats: int) -> None:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim")
    grid.add_column()
    grid.add_column(justify="right", style="green")
    grid.add_row("[semantic]", "papers fetched", str(count_a))
    grid.add_row("[structural]", "papers fetched", str(count_b))
    grid.add_row("Total:", f"[bold]{count_a + count_b}[/bold] papers", f"{n_cats} arXiv categories")
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


def _results_table(pairs: list[tuple], top_n: int) -> None:
    if not pairs:
        console.print(
            Panel(
                "[yellow]No cross-domain pairs found.[/yellow]\n\n"
                "This may mean both queries retrieved papers from the same arXiv categories, "
                "or the embedding model needs a stronger structural query.\n\n"
                "Try providing [cyan]--structural-query[/cyan] with explicit cross-domain terms.",
                title="[bold]Results[/bold]",
                border_style="yellow",
            )
        )
        return

    table = Table(
        title=f"Top {min(top_n, len(pairs))} Cross-Domain Similar Pairs",
        border_style="green",
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Score", width=6, justify="right")
    table.add_column("Category", width=12)
    table.add_column("Title", overflow="fold")

    for rank, (id1, title1, cat1, id2, title2, cat2, score) in enumerate(pairs, start=1):
        t1 = title1[:80] + ("…" if len(title1) > 80 else "")
        t2 = title2[:80] + ("…" if len(title2) > 80 else "")
        table.add_row(
            str(rank),
            f"{score:.4f}",
            f"[cyan]{cat1}[/cyan]",
            f"[bold]{t1}[/bold]\n[dim]{id1}[/dim]",
        )
        table.add_row("", "", f"[yellow]{cat2}[/yellow]", f"{t2}\n[dim]{id2}[/dim]")

    console.print(table)


def _interpretation(pairs: list[tuple]) -> None:
    if not pairs:
        return
    _, _, cat1, _, _, cat2, score = pairs[0]
    console.print(
        f"\n  [bold green]{len(pairs)} cross-domain pair(s) found.[/bold green]"
        f"  Top match: [cyan]{cat1}[/cyan] ↔ [yellow]{cat2}[/yellow]"
        f"  (score [bold]{score:.4f}[/bold])\n"
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

            cls_color = {
                "STRUCTURAL": "green",
                "METHODOLOGICAL": "yellow",
                "THEMATIC": "blue",
            }.get(pair.classification, "white")

            console.print(f"  [bold]{i}.[/bold] [cyan]{title_a}[/cyan]")
            console.print(f"     [dim]arxiv:{pair.paper_a}[/dim]")
            console.print(f"     ↔ [yellow]{title_b}[/yellow]")
            console.print(f"     [dim]arxiv:{pair.paper_b}[/dim]")
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


# ── CLI commands ──────────────────────────────────────────────────────────────

@click.group()
@click.version_option(__version__, prog_name="arxanon")
def main() -> None:
    """Arxanon — cross-domain structural analogy discovery for AI/ML researchers.

    Finds papers from fields you don't follow that describe the exact mathematical
    structure you're stuck on, and tells you precisely why they matter.
    """


@main.command()
@click.argument("query")
@click.option(
    "--structural-query",
    default=None,
    metavar="QUERY",
    help=(
        "Query targeting papers from other domains with similar mathematical structure. "
        "If omitted, a generic structural companion query is derived automatically."
    ),
)
@click.option(
    "--max-results",
    default=100,
    show_default=True,
    metavar="N",
    help="Maximum papers to fetch per query.",
)
@click.option(
    "--top-n",
    default=5,
    show_default=True,
    metavar="N",
    help="Number of top cross-domain pairs to display.",
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
def search(
    query: str,
    structural_query: Optional[str],
    max_results: int,
    top_n: int,
    no_gemma: bool,
    no_tda: bool,
    coupling_threshold: int,
    top_clusters: int,
) -> None:
    """Search for cross-domain structural analogies to QUERY.

    Runs the full Phase 1 pipeline (retrieval + citation graph + embeddings),
    then Phase 2 (bridge detection via citation exclusion + HDBSCAN), and
    optionally Phase 3 (Gemma 4 structural analogy verification via Ollama).

    Example:

        arxanon search "edge of stability gradient descent neural networks"

        arxanon search "gradient descent oscillation" --no-gemma --no-tda
    """
    if structural_query is None:
        structural_query = default_structural_query(query)

    _header()
    init_db()

    # ── Phase 1, Stage 1: Fetch papers ────────────────────────────────────────
    console.print("[bold]Stage 1:[/bold] Retrieving papers from arXiv\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_a = progress.add_task(f"[cyan][semantic][/cyan]  {query[:60]}", total=max_results)
        count_a = fetch_and_store_papers(
            query,
            max_results,
            "semantic",
            on_paper=lambda n: progress.update(task_a, completed=n),
        )
        progress.update(
            task_a,
            completed=count_a,
            description=f"[green]✓[/green] [semantic]   {query[:60]}",
        )

        task_b = progress.add_task(
            f"[yellow][structural][/yellow] {structural_query[:60]}", total=max_results
        )
        count_b = fetch_and_store_papers(
            structural_query,
            max_results,
            "structural",
            on_paper=lambda n: progress.update(task_b, completed=n),
        )
        progress.update(
            task_b,
            completed=count_b,
            description=f"[green]✓[/green] [structural] {structural_query[:60]}",
        )

    n_cats = get_unique_category_count()
    _retrieval_panel(count_a, count_b, n_cats)

    # ── Phase 1, Stage 2: Citation graph ──────────────────────────────────────
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
    _citation_panel(edge_counts.get("direct", 0), edge_counts.get("cocitation", 0))

    # ── Phase 1, Stage 3: Embeddings ──────────────────────────────────────────
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
        index_size = embed_and_index_papers()
        progress.update(
            task_e, description=f"[green]✓[/green] {index_size} vectors indexed"
        )

    _embedding_panel(index_size, config.EMBED_MODEL)

    # ── Phase 1, Stage 4: Cross-domain pairs ──────────────────────────────────
    console.print("[bold]Stage 4:[/bold] Finding cross-domain similar pairs\n")

    pairs = find_top_similar_pairs(n=top_n, cross_only=True)
    _results_table(pairs, top_n)
    _interpretation(pairs)

    if index_size == 0:
        return

    # ── Phase 2: Bridge detection ─────────────────────────────────────────────
    console.print("[bold]Stage 5:[/bold] Detecting citation-isolated bridge clusters\n")

    from .bridge_pipeline import run_bridge_pipeline, run_gemma_validation

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

        bridge_result = run_bridge_pipeline(
            coupling_threshold=coupling_threshold,
            top_n_clusters=top_clusters,
            enable_tda=not no_tda,
            on_stage=_on_bridge_stage,
        )

    papers_embedded = load_papers_with_embeddings()
    _bridge_detection_panel(bridge_result)

    # ── Phase 3: Gemma validation ─────────────────────────────────────────────
    if not no_gemma:
        console.print(
            "[bold]Stage 6:[/bold] Gemma 4 structural analogy verification\n"
        )

        total_pairs_estimate = sum(
            len(c.bridge_edges) for c in bridge_result.clusters[:top_clusters]
        )

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
                on_pair=_on_pair,
            )
            if bridge_result.gemma_available:
                progress.update(task_g, description="[green]✓[/green] Validation complete")
            else:
                progress.update(task_g, description="[yellow]~[/yellow] Ollama unavailable")

        _gemma_validation_panel(bridge_result)

    # ── Display bridge clusters ───────────────────────────────────────────────
    if bridge_result.clusters:
        _bridge_clusters_table(bridge_result.clusters[:top_clusters])
        _bridge_cluster_detail(bridge_result.clusters[0], papers_embedded)
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


if __name__ == "__main__":
    sys.exit(main())
