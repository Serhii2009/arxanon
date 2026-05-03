"""Interactive terminal session: setup wizard + query loop."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from . import __version__, config

_SETTINGS_PATH = config.DATA_DIR / "settings.json"
_DEFAULTS: dict = {
    "embed_model": "BAAI/bge-large-en-v1.5",
    "gemma_model": "gemma4:e2b",
    "openrouter_api_key": "",
    "max_results": 100,
    "max_validate": 5,
    "top_clusters": 3,
    "coupling_threshold": 3,
    "setup_done": False,
}

# Validation bounds for numeric settings
_BOUNDS: dict[str, tuple[int, int]] = {
    "max_results": (10, 500),
    "max_validate": (1, 100),
    "top_clusters": (1, 10),
    "coupling_threshold": (1, 20),
}


# ── Settings persistence ──────────────────────────────────────────────────────

def load_settings() -> dict:
    try:
        if _SETTINGS_PATH.exists():
            data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
            return {**_DEFAULTS, **data}
    except Exception:
        pass
    return dict(_DEFAULTS)


def save_settings(settings: dict) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _SETTINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    tmp.replace(_SETTINGS_PATH)


# ── Input helpers ─────────────────────────────────────────────────────────────

def _prompt_int(
    console: Console,
    prompt: str,
    current: int,
    min_val: int,
    max_val: int,
) -> int:
    """Prompt for an integer in [min_val, max_val]. Returns current on empty input or invalid."""
    while True:
        try:
            raw = console.input(f"  {prompt} [{min_val}–{max_val}, current: {current}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return current
        if not raw:
            return current
        try:
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            console.print(f"  [yellow]Must be between {min_val} and {max_val}.[/yellow]")
        except ValueError:
            console.print("  [yellow]Enter a whole number.[/yellow]")


# ── Setup wizard ──────────────────────────────────────────────────────────────

def run_setup_wizard(console: Console) -> dict:
    console.print(
        Panel(
            Text.assemble(
                ("Welcome to arxanon!\n\n", "bold white"),
                ("This wizard runs once to configure your environment.\n", "dim"),
                ("Settings are saved to ~/.arxanon/settings.json", "dim"),
            ),
            border_style="cyan",
            padding=(0, 2),
        )
    )

    settings = dict(_DEFAULTS)

    # Embedding model
    console.print("\n[bold]Step 1: Embedding model[/bold]")
    console.print("  [cyan]1[/cyan]  BAAI/bge-large-en-v1.5  [dim](CPU-friendly, 1024-dim — recommended for laptops)[/dim]")
    console.print("  [cyan]2[/cyan]  nvidia/NV-Embed-v2       [dim](GPU required, 4096-dim — higher quality)[/dim]")
    try:
        choice = console.input("\n  Choice [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Setup cancelled — using defaults.[/yellow]")
        settings["setup_done"] = True
        save_settings(settings)
        return settings

    settings["embed_model"] = "nvidia/NV-Embed-v2" if choice == "2" else "BAAI/bge-large-en-v1.5"

    # Gemma model
    console.print("\n[bold]Step 2: Gemma model[/bold]")
    available_gemma = _list_ollama_gemma_models()

    if available_gemma:
        for i, name in enumerate(available_gemma, start=1):
            console.print(f"  [cyan]{i}[/cyan]  {name}")
        console.print("  [cyan]c[/cyan]  Enter custom model name")
        try:
            choice = console.input("\n  Choice [1]: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            settings["gemma_model"] = available_gemma[0]
            choice = "done"

        if choice != "done":
            if choice.lower() == "c":
                try:
                    settings["gemma_model"] = console.input("  Model name: ").strip() or available_gemma[0]
                except (EOFError, KeyboardInterrupt):
                    settings["gemma_model"] = available_gemma[0]
            else:
                try:
                    idx = int(choice) - 1
                    settings["gemma_model"] = available_gemma[max(0, min(idx, len(available_gemma) - 1))]
                except ValueError:
                    settings["gemma_model"] = available_gemma[0]
    else:
        console.print("  [yellow]No Ollama models found.[/yellow]")
        console.print("  [dim]Run: ollama serve && ollama pull gemma4:e2b[/dim]")
        try:
            name = console.input("  Model name [gemma4:e2b]: ").strip()
        except (EOFError, KeyboardInterrupt):
            name = ""
        settings["gemma_model"] = name or "gemma4:e2b"

    # Pipeline settings
    console.print("\n[bold]Step 3: Pipeline settings[/bold]")
    console.print("  [dim]Press Enter to accept defaults.[/dim]\n")
    settings["max_results"] = _prompt_int(console, "Papers fetched per query (max_results)", 100, 10, 500)
    settings["max_validate"] = _prompt_int(console, "Bridge pairs sent to Gemma (max_validate)", 5, 1, 100)
    settings["top_clusters"] = _prompt_int(console, "Clusters shown in output (top_clusters)", 3, 1, 10)
    settings["coupling_threshold"] = _prompt_int(console, "Bibcoupling threshold (coupling_threshold)", 3, 1, 20)

    settings["setup_done"] = True
    save_settings(settings)

    embed_label = settings["embed_model"].split("/")[-1]
    console.print(
        f"\n  [green]✓[/green] Settings saved:"
        f" embedding=[cyan]{embed_label}[/cyan]"
        f"  gemma=[cyan]{settings['gemma_model']}[/cyan]"
        f"  results=[cyan]{settings['max_results']}[/cyan]"
        f"  validate=[cyan]{settings['max_validate']}[/cyan]\n"
    )
    return settings


def _list_ollama_gemma_models() -> list[str]:
    try:
        import requests
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        gemma = [n for n in names if "gemma" in n.lower()]
        return gemma or names[:5]
    except Exception:
        return []


def _list_ollama_models() -> list[str]:
    try:
        import requests
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        resp.raise_for_status()
        return [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        return []


# ── /settings handler ─────────────────────────────────────────────────────────

def _handle_settings(console: Console, settings: dict) -> dict:
    embed_label = settings["embed_model"].split("/")[-1]
    if settings.get("openrouter_api_key"):
        provider_label = f"OpenRouter · {settings['gemma_model']}"
    else:
        provider_label = f"Ollama · {settings['gemma_model']}"
    console.print(
        f"\n  [bold]Current settings[/bold]\n"
        f"  [dim]1[/dim]  Embedding model:    [cyan]{embed_label}[/cyan]\n"
        f"  [dim]2[/dim]  LLM provider/model: [cyan]{provider_label}[/cyan]\n"
        f"  [dim]3[/dim]  max_results:        [cyan]{settings['max_results']}[/cyan]"
        f"  [dim](papers fetched per query)[/dim]\n"
        f"  [dim]4[/dim]  max_validate:       [cyan]{settings['max_validate']}[/cyan]"
        f"  [dim](bridge pairs sent to Gemma)[/dim]\n"
        f"  [dim]5[/dim]  top_clusters:       [cyan]{settings['top_clusters']}[/cyan]"
        f"  [dim](clusters shown in output)[/dim]\n"
        f"  [dim]6[/dim]  coupling_threshold: [cyan]{settings['coupling_threshold']}[/cyan]"
        f"  [dim](bibcoupling filter)[/dim]\n"
    )
    console.print("  Enter a number to change it, or press Enter to cancel.\n")

    try:
        choice = console.input("  Setting to change [1–6]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return settings

    if choice == "1":
        console.print("\n  [cyan]1[/cyan] BAAI/bge-large-en-v1.5   [cyan]2[/cyan] nvidia/NV-Embed-v2")
        try:
            c = console.input(f"  Choice (current: {embed_label}): ").strip()
        except (EOFError, KeyboardInterrupt):
            c = ""
        if c == "1":
            settings["embed_model"] = "BAAI/bge-large-en-v1.5"
        elif c == "2":
            settings["embed_model"] = "nvidia/NV-Embed-v2"

    elif choice == "2":
        console.print("\n  [bold]LLM provider[/bold]")
        console.print("  [cyan]1[/cyan]  Ollama  [dim](local, free)[/dim]")
        console.print("  [cyan]2[/cyan]  OpenRouter  [dim](cloud, requires API key)[/dim]")
        try:
            provider_choice = console.input("\n  Provider [1]: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            provider_choice = ""

        if provider_choice == "2":
            try:
                raw_key = console.input("  OpenRouter API key: ", password=True).strip()
            except (EOFError, KeyboardInterrupt):
                raw_key = ""
            if raw_key:
                settings["openrouter_api_key"] = raw_key
                os.environ["OPENROUTER_API_KEY"] = raw_key
                config.OPENROUTER_API_KEY = raw_key
                config.USE_OPENROUTER = True

            or_models = [
                "google/gemma-2-27b-it",
                "google/gemma-3-27b-it",
                "google/gemma-2-9b-it",
            ]
            console.print("\n  [bold]OpenRouter model[/bold]")
            for i, m in enumerate(or_models, 1):
                console.print(f"  [cyan]{i}[/cyan]  {m}")
            console.print("  [cyan]c[/cyan]  Enter custom model ID")
            try:
                mc = console.input(f"\n  Choice (current: {settings['gemma_model']}): ").strip()
            except (EOFError, KeyboardInterrupt):
                mc = ""
            if mc == "c":
                try:
                    custom = console.input("  Model ID: ").strip()
                except (EOFError, KeyboardInterrupt):
                    custom = ""
                if custom:
                    settings["gemma_model"] = custom
            elif mc.isdigit():
                idx = int(mc) - 1
                if 0 <= idx < len(or_models):
                    settings["gemma_model"] = or_models[idx]

        elif provider_choice == "1":
            settings["openrouter_api_key"] = ""
            os.environ.pop("OPENROUTER_API_KEY", None)
            config.OPENROUTER_API_KEY = ""
            config.USE_OPENROUTER = False

            available = _list_ollama_models()
            if available:
                console.print("\n  [bold]Ollama models[/bold]")
                for i, name in enumerate(available, 1):
                    console.print(f"  [cyan]{i}[/cyan]  {name}")
                console.print("  [cyan]c[/cyan]  Enter custom model name")
                try:
                    mc = console.input(f"\n  Choice (current: {settings['gemma_model']}): ").strip()
                except (EOFError, KeyboardInterrupt):
                    mc = ""
                if mc == "c":
                    try:
                        custom = console.input("  Model name: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        custom = ""
                    if custom:
                        settings["gemma_model"] = custom
                elif mc.isdigit():
                    idx = int(mc) - 1
                    if 0 <= idx < len(available):
                        settings["gemma_model"] = available[idx]
            else:
                console.print(f"\n  [yellow]No Ollama models found.[/yellow]")
                console.print(f"  [dim]Run: ollama serve && ollama pull {settings['gemma_model']}[/dim]")
                try:
                    custom = console.input(f"  Model name (current: {settings['gemma_model']}): ").strip()
                except (EOFError, KeyboardInterrupt):
                    custom = ""
                if custom:
                    settings["gemma_model"] = custom

    elif choice == "3":
        settings["max_results"] = _prompt_int(
            console, "max_results", settings["max_results"], *_BOUNDS["max_results"]
        )
    elif choice == "4":
        settings["max_validate"] = _prompt_int(
            console, "max_validate", settings["max_validate"], *_BOUNDS["max_validate"]
        )
    elif choice == "5":
        settings["top_clusters"] = _prompt_int(
            console, "top_clusters", settings["top_clusters"], *_BOUNDS["top_clusters"]
        )
    elif choice == "6":
        settings["coupling_threshold"] = _prompt_int(
            console, "coupling_threshold", settings["coupling_threshold"], *_BOUNDS["coupling_threshold"]
        )

    save_settings(settings)
    os.environ["ARXANON_EMBED_MODEL"] = settings["embed_model"]
    os.environ["ARXANON_GEMMA_MODEL"] = settings["gemma_model"]
    config.EMBED_MODEL = settings["embed_model"]
    config.GEMMA_MODEL = settings["gemma_model"]
    console.print(f"\n  [green]✓[/green] Settings updated.\n")
    return settings


# ── Interactive session ───────────────────────────────────────────────────────

def run_interactive_session() -> None:
    console = Console()
    settings = load_settings()

    if not settings.get("setup_done"):
        settings = run_setup_wizard(console)

    # Apply model settings to live config (module already loaded before env var takes effect)
    os.environ["ARXANON_EMBED_MODEL"] = settings["embed_model"]
    os.environ["ARXANON_GEMMA_MODEL"] = settings["gemma_model"]
    config.EMBED_MODEL = settings["embed_model"]
    config.GEMMA_MODEL = settings["gemma_model"]

    api_key = settings.get("openrouter_api_key", "")
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
        config.OPENROUTER_API_KEY = api_key
        config.USE_OPENROUTER = True

    _print_session_header(console, settings)

    from .db import init_db
    init_db()

    while True:
        try:
            raw = console.input("\n[bold cyan]  Research problem >[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n\n  [dim]Goodbye.[/dim]")
            break

        if not raw:
            continue
        if raw.lower() in ("/quit", "/exit", "/q"):
            console.print("\n  [dim]Goodbye.[/dim]")
            break
        if raw.lower() == "/help":
            _print_help(console)
            continue
        if raw.lower() == "/settings":
            settings = _handle_settings(console, settings)
            _print_session_header(console, settings)
            continue

        _run_search(raw, console, settings)


def _print_result_panel(
    bridge_result: object,
    papers: dict,
    out_dir: Path,
    console: Console,
) -> None:
    from .clusters import BridgePipelineResult
    assert isinstance(bridge_result, BridgePipelineResult)

    best_cluster = bridge_result.clusters[0] if bridge_result.clusters else None
    if not best_cluster:
        return

    cats = best_cluster.categories
    domain_a = cats[0] if cats else "?"
    domain_b = cats[1] if len(cats) > 1 else "?"

    explanation = ""
    top_paper_title = ""
    top_paper_id = ""

    if best_cluster.validated_pairs:
        pair = best_cluster.validated_pairs[0]
        chain = (pair.reasoning_chain or "").strip()
        sents = [s.strip() for s in chain.replace("\n", " ").split(". ") if len(s.strip()) > 15]
        explanation = ". ".join(sents[:3])
        if explanation and not explanation.endswith("."):
            explanation += "."
        pb = papers.get(pair.paper_b, {})
        top_paper_title = (pb.get("title") or pair.paper_b)[:70]
        top_paper_id = pair.paper_b
    elif best_cluster.bridge_edges:
        id_a, id_b, _ = best_cluster.bridge_edges[0]
        explanation = (
            f"Papers from {domain_a} and {domain_b} share similar mathematical structure. "
            "Run with Gemma validation to see the structural explanation."
        )
        pb = papers.get(id_b, {})
        top_paper_title = (pb.get("title") or id_b)[:70]
        top_paper_id = id_b

    # Fall back to highest query-relevance paper if needed
    if not top_paper_id:
        qrs = bridge_result.query_relevance_scores
        if qrs:
            best_pid = max(qrs, key=qrs.__getitem__)
            bp = papers.get(best_pid, {})
            top_paper_title = (bp.get("title") or best_pid)[:70]
            top_paper_id = best_pid

    parts = [
        f"[bold]Strong connection:[/bold] [cyan]{domain_a}[/cyan] ↔ [yellow]{domain_b}[/yellow]",
        "",
    ]
    if explanation:
        parts.append(explanation)
        parts.append("")
    if top_paper_title:
        parts += [
            "[bold]Top paper to read:[/bold]",
            f"[italic]{top_paper_title}[/italic]",
            f"[dim]arxiv:{top_paper_id}[/dim]",
            "",
        ]
    parts.append(f"[dim]Full analysis → ./{out_dir.name}/[/dim]")

    console.print(
        Panel(
            "\n".join(parts),
            title="[bold]What I found[/bold]",
            border_style="cyan",
            padding=(0, 2),
        )
    )


def _gemma_synthesis_panel(
    query: str,
    bridge_result: object,
    papers: dict,
    out_dir: Path,
    console: Console,
) -> None:
    from .clusters import BridgePipelineResult
    from .llm_client import call_llm
    assert isinstance(bridge_result, BridgePipelineResult)

    # Top 5 papers by query relevance
    qrs = bridge_result.query_relevance_scores or {}
    top_pids = sorted(qrs, key=qrs.__getitem__, reverse=True)[:5]
    paper_lines: list[str] = []
    for pid in top_pids:
        p = papers.get(pid, {})
        title = p.get("title", pid)
        abstract = (p.get("abstract") or "")[:300]
        try:
            cats = json.loads(p.get("categories", "[]") or "[]")
            cat = cats[0] if cats else "?"
        except Exception:
            cat = "?"
        paper_lines.append(f"- {title} (arxiv:{pid}, {cat})\n  {abstract}")
    papers_text = "\n".join(paper_lines) if paper_lines else "No papers retrieved."

    # Bridge findings
    all_validated = [p for c in bridge_result.clusters for p in c.validated_pairs]
    if all_validated:
        bridge_lines: list[str] = []
        for pair in all_validated[:3]:
            pa = papers.get(pair.paper_a, {})
            pb = papers.get(pair.paper_b, {})
            ta = pa.get("title", pair.paper_a)
            tb = pb.get("title", pair.paper_b)
            bridge_lines.append(
                f"- {pair.classification}: arxiv:{pair.paper_a} ({ta[:60]}) "
                f"↔ arxiv:{pair.paper_b} ({tb[:60]})"
            )
            if pair.reasoning_chain and pair.reasoning_chain != "[No reasoning captured]":
                bridge_lines.append(f"  Reasoning: {pair.reasoning_chain[:200]}")
        bridge_text = "\n".join(bridge_lines)
    elif bridge_result.clusters:
        best = bridge_result.clusters[0]
        cats_str = " ↔ ".join(best.categories[:3]) if best.categories else "?"
        bridge_text = (
            f"Bridge clusters found ({cats_str}) but Gemma validation produced no confirmed pairs. "
            "Connections are based on embedding similarity only."
        )
    else:
        bridge_text = "No cross-domain bridges found — all papers appear to be from the same domain."

    prompt = (
        f'You are analyzing papers found for this research query: "{query}"\n\n'
        f"Top papers by relevance:\n{papers_text}\n\n"
        f"Bridge detector findings:\n{bridge_text}\n\n"
        "Answer exactly these three questions in plain language, grounded only in what "
        "was actually found above. Do not invent papers, results, or claims.\n\n"
        "1. What do the most interesting papers show?\n"
        "2. Is there any unexpected cross-domain connection in the findings?\n"
        "3. What is one concrete, mathematically specific next step a researcher could pursue?\n\n"
        "Keep each answer to 2-3 sentences. No labels or headers — just three paragraphs."
    )

    gemma_text = ""
    try:
        gemma_text = call_llm(prompt, timeout=30, temperature=0.3)
    except Exception:
        pass

    if not gemma_text:
        if top_pids:
            p = papers.get(top_pids[0], {})
            title = p.get("title", top_pids[0])
            gemma_text = (
                f"[dim](LLM unavailable — run: ollama serve && ollama pull {config.GEMMA_MODEL})[/dim]\n\n"
                f"Top result: {title}\n[dim]arxiv:{top_pids[0]}[/dim]"
            )
        else:
            gemma_text = "[dim](No results found)[/dim]"

    console.print(
        Panel(
            gemma_text + f"\n\n[dim]Full analysis → ./{out_dir.name}/[/dim]",
            title="[bold]What I found[/bold]",
            border_style="cyan",
            padding=(0, 2),
        )
    )


def _run_search(query: str, console: Console, settings: dict) -> None:
    from .cli import execute_pipeline
    from .output_writer import save_session

    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
    if config.FAISS_PATH.exists():
        config.FAISS_PATH.unlink()

    from .db import init_db
    init_db()

    max_validate: Optional[int] = settings.get("max_validate")

    result = execute_pipeline(
        query=query,
        max_results=settings.get("max_results", 100),
        top_clusters=settings.get("top_clusters", 3),
        max_validate=max_validate,
        no_gemma=False,
        no_tda=False,
        coupling_threshold=settings.get("coupling_threshold", 3),
        verbose=False,
    )

    if result:
        bridge_result, papers = result
        out_dir = save_session(query, bridge_result, papers)
        _gemma_synthesis_panel(query, bridge_result, papers, out_dir, console)


def _print_session_header(console: Console, settings: dict) -> None:
    embed_label = settings["embed_model"].split("/")[-1]
    if settings.get("openrouter_api_key"):
        llm_label = f"OpenRouter · {settings['gemma_model']}"
    else:
        llm_label = f"Ollama · {settings['gemma_model']}"
    header = Text.assemble(
        ("A R X A N O N", "bold white"),
        (f"  v{__version__}", "dim white"),
        "\n",
        ("Cross-Domain Structural Analogy Engine", "dim cyan"),
    )
    console.print(Panel(header, border_style="cyan", padding=(0, 2)))
    console.print(
        f"  Embedding: [cyan]{embed_label}[/cyan]"
        f"  ·  LLM: [cyan]{llm_label}[/cyan]\n"
        f"  Results: [cyan]{settings.get('max_results', 100)}[/cyan]"
        f"  ·  Validate: [cyan]{settings.get('max_validate', 5)}[/cyan]"
        f"  ·  Clusters: [cyan]{settings.get('top_clusters', 3)}[/cyan]"
        f"  ·  [dim]/help for commands[/dim]\n"
    )


def _print_help(console: Console) -> None:
    console.print(
        Panel(
            "  /help      Show this message\n"
            "  /settings  View and change any of the 6 settings\n"
            "  /quit      Exit the session\n\n"
            "  [dim]Or just type your research problem and press Enter.[/dim]",
            title="[bold]Commands[/bold]",
            border_style="dim",
            padding=(0, 2),
        )
    )
