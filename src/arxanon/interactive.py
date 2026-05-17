"""Interactive terminal session: setup wizard + query loop."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

import questionary
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style as PromptStyle
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__, config

_SETTINGS_PATH = config.DATA_DIR / "settings.json"
_DEFAULTS: dict = {
    "embed_model": "BAAI/bge-large-en-v1.5",
    "gemma_model": "gemma4:e2b",
    "openrouter_api_key": "",
    "max_results": 100,
    "max_validate": 50,
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

_SLASH_COMMANDS = [
    ("/help",     "Show all available commands"),
    ("/clear",    "Clear the screen and reprint the session header"),
    ("/history",  "Show previous queries this session"),
    ("/save",     "Save the last report files to a named location"),
    ("/rerun",    "Re-run the most recent query"),
    ("/settings", "Open the settings interface"),
    ("/fields",   "Show which scientific fields were reached in the last run"),
    ("/pairs",    "Show all validated connections from the last run"),
    ("/quit",     "Exit Arxanon"),
]


class _SlashCompleter(Completer):
    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        word = text.lower()
        for cmd, desc in _SLASH_COMMANDS:
            if cmd.startswith(word):
                yield Completion(cmd, start_position=-len(text), display_meta=desc)


_PROMPT_STYLE = PromptStyle.from_dict({
    "completion-menu.completion.current": "bg:ansipurple fg:ansiwhite bold",
    "completion-menu.completion":         "bg:ansibrightblack fg:ansiwhite",
})

_PROMPT_MSG = FormattedText([("", "\n"), ("bold fg:ansicyan", "  Research problem > ")])

_Q_STYLE = questionary.Style([
    ("selected",    "fg:purple bold"),
    ("pointer",     "fg:purple bold"),
    ("highlighted", "fg:purple bold"),
])


# ── Settings persistence ──────────────────────────────────────────────────────

def load_settings() -> dict:
    try:
        if _SETTINGS_PATH.exists():
            data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
            merged = {**_DEFAULTS, **data}
            if merged.get("max_validate", 0) < 10:
                merged["max_validate"] = _DEFAULTS["max_validate"]
            return merged
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
    settings["max_validate"] = _prompt_int(console, "Bridge pairs sent to Gemma (max_validate)", settings["max_validate"], 1, 100)
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

def _settings_openrouter(console: Console, working: dict) -> bool:
    model_choice = questionary.select(
        "Choose Gemma 4 model:",
        choices=[
            "google/gemma-4-31b-it    (best quality)",
            "google/gemma-4-9b-it     (faster)",
            "Enter custom model ID",
            "← Back",
        ],
        style=_Q_STYLE,
    ).ask()

    if model_choice is None or model_choice == "← Back":
        return False

    if model_choice == "Enter custom model ID":
        custom = questionary.text("Model ID:").ask()
        if not custom:
            return False
        working["gemma_model"] = custom.strip()
    elif "31b" in model_choice:
        working["gemma_model"] = "google/gemma-4-31b-it"
    else:
        working["gemma_model"] = "google/gemma-4-9b-it"

    raw_key = questionary.password(
        "OpenRouter API key (press Enter to keep existing key):"
    ).ask()
    if raw_key is None:
        return False
    if raw_key:
        working["openrouter_api_key"] = raw_key

    return True


def _settings_ollama(console: Console, working: dict) -> bool:
    try:
        import requests
        resp = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        resp.raise_for_status()
        available = [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        console.print(
            f"  [yellow]Could not connect to Ollama at {config.OLLAMA_BASE_URL}.\n"
            "  Make sure Ollama is running and try again.[/yellow]"
        )
        return False

    ollama_choices = available + ["Enter custom model name", "← Back"]
    model_choice = questionary.select(
        "Choose local model:",
        choices=ollama_choices,
        style=_Q_STYLE,
    ).ask()

    if model_choice is None or model_choice == "← Back":
        return False

    if model_choice == "Enter custom model name":
        custom = questionary.text("Model name:").ask()
        if not custom:
            return False
        working["gemma_model"] = custom.strip()
    else:
        working["gemma_model"] = model_choice

    url = questionary.text(
        "Ollama server URL:",
        default=config.OLLAMA_BASE_URL,
    ).ask()
    if url is None:
        return False
    config.OLLAMA_BASE_URL = url.strip()

    working["openrouter_api_key"] = ""
    return True


def _settings_llm(console: Console, working: dict) -> bool:
    provider = questionary.select(
        "Choose provider:",
        choices=[
            "OpenRouter  (cloud, requires API key)",
            "Ollama      (local, free, requires Ollama running)",
            "← Back",
        ],
        style=_Q_STYLE,
    ).ask()

    if provider is None or provider == "← Back":
        return False
    if provider.startswith("OpenRouter"):
        return _settings_openrouter(console, working)
    return _settings_ollama(console, working)


def _handle_settings(console: Console, settings: dict) -> dict:
    embed_label = settings["embed_model"].split("/")[-1]
    provider_label = (
        f"OpenRouter · {settings['gemma_model']}"
        if settings.get("openrouter_api_key") else
        f"Ollama · {settings['gemma_model']}"
    )
    console.print(
        f"\n  [bold]Current settings[/bold]\n"
        f"  Embedding model:    [cyan]{embed_label}[/cyan]\n"
        f"  LLM provider/model: [cyan]{provider_label}[/cyan]\n"
        f"  max_results:        [cyan]{settings['max_results']}[/cyan]"
        f"  [dim](papers fetched per query)[/dim]\n"
        f"  max_validate:       [cyan]{settings['max_validate']}[/cyan]"
        f"  [dim](bridge pairs sent to Gemma)[/dim]\n"
        f"  top_clusters:       [cyan]{settings['top_clusters']}[/cyan]"
        f"  [dim](clusters shown in output)[/dim]\n"
        f"  coupling_threshold: [cyan]{settings['coupling_threshold']}[/cyan]"
        f"  [dim](bibcoupling filter)[/dim]\n"
    )

    working = dict(settings)

    choice = questionary.select(
        "What would you like to change?",
        choices=[
            "LLM provider and model",
            f"Results per query (currently: {working['max_results']})",
            f"Pairs to validate (currently: {working['max_validate']})",
            "Done",
        ],
        style=_Q_STYLE,
    ).ask()

    if choice is None or choice == "Done":
        return settings

    changed = False
    if choice == "LLM provider and model":
        changed = _settings_llm(console, working)
    elif choice.startswith("Results per query"):
        working["max_results"] = _prompt_int(
            console, "max_results", working["max_results"], *_BOUNDS["max_results"]
        )
        changed = True
    elif choice.startswith("Pairs to validate"):
        working["max_validate"] = _prompt_int(
            console, "max_validate", working["max_validate"], *_BOUNDS["max_validate"]
        )
        changed = True

    if not changed:
        return settings

    save_settings(working)
    os.environ["ARXANON_EMBED_MODEL"] = working["embed_model"]
    os.environ["ARXANON_GEMMA_MODEL"] = working["gemma_model"]
    config.EMBED_MODEL = working["embed_model"]
    config.GEMMA_MODEL = working["gemma_model"]
    if working.get("openrouter_api_key"):
        os.environ["OPENROUTER_API_KEY"] = working["openrouter_api_key"]
        config.OPENROUTER_API_KEY = working["openrouter_api_key"]
        config.USE_OPENROUTER = True
    else:
        os.environ.pop("OPENROUTER_API_KEY", None)
        config.OPENROUTER_API_KEY = ""
        config.USE_OPENROUTER = False
    console.print("\n  [green]✓[/green] Settings updated.\n")
    return working


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

    _query_history: list[str] = []
    _last: list = [None]  # _last[0] = (bridge_result, papers, out_dir) | None

    def _run_and_store(q: str) -> None:
        r = _run_search(q, console, settings)
        if r is not None:
            _query_history.append(q)
            _last[0] = r

    _session = PromptSession(
        completer=_SlashCompleter(),
        style=_PROMPT_STYLE,
        complete_while_typing=True,
    )

    while True:
        try:
            raw = _session.prompt(_PROMPT_MSG).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n\n  [dim]Goodbye.[/dim]")
            break

        if not raw:
            continue

        low = raw.lower()
        if low in ("/quit", "/exit", "/q"):
            console.print("\n  [dim]Goodbye.[/dim]")
            break
        elif low == "/help":
            _cmd_help(console)
        elif low == "/clear":
            _cmd_clear(console, settings)
        elif low == "/history":
            _cmd_history(console, _query_history, _run_and_store)
        elif low == "/rerun":
            if _query_history:
                console.print(f"\n  Re-running: [italic]{_query_history[-1]}[/italic]\n")
                r = _run_search(_query_history[-1], console, settings)
                if r is not None:
                    _last[0] = r
            else:
                console.print("  [yellow]No previous query.[/yellow]")
        elif low == "/save" or low.startswith("/save "):
            _cmd_save(console, _last[0], raw[5:].strip() or None)
        elif low == "/fields":
            _cmd_fields(console, _last[0])
        elif low == "/pairs":
            _cmd_pairs(console, _last[0])
        elif low == "/settings":
            settings = _handle_settings(console, settings)
            _print_session_header(console, settings)
        else:
            _run_and_store(raw)


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
    directions: Optional[str] = None,
) -> None:
    from .clusters import BridgePipelineResult
    assert isinstance(bridge_result, BridgePipelineResult)

    direct_pairs_all = bridge_result.direct_cross_domain_pairs or []
    strong_direct = [p for p in direct_pairs_all if p.classification == "STRUCTURAL"]
    related_direct = [p for p in direct_pairs_all if p.classification in ("METHODOLOGICAL", "THEMATIC")]
    all_direct = strong_direct + related_direct

    if directions is not None:
        gemma_text = directions
        llm_failed = False
        qrs_missing = False
    else:
        from .llm_client import call_llm

        qrs = bridge_result.query_relevance_scores or {}
        qrs_missing = not qrs and bool(papers)
        top_pids = sorted(qrs, key=qrs.__getitem__, reverse=True)[:5]
        if not top_pids and papers:
            str_pids = [pid for pid, p in papers.items() if (p.get("query_tag") or "").startswith("str")]
            sem_pids = [pid for pid, p in papers.items() if (p.get("query_tag") or "").startswith("sem")]
            top_pids = (str_pids[:3] + sem_pids[:2])[:5]
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

        all_validated = [p for c in bridge_result.clusters for p in c.validated_pairs]

        if all_validated or all_direct:
            best_pairs = all_direct[:3] if all_direct else all_validated[:3]
            bridge_lines: list[str] = []
            for pair in best_pairs:
                pa = papers.get(pair.paper_a, {})
                pb = papers.get(pair.paper_b, {})
                ta = pa.get("title", pair.paper_a)
                tb = pb.get("title", pair.paper_b)
                try:
                    cats = json.loads(pb.get("categories", "[]") or "[]")
                    cat_b = cats[0] if cats else "?"
                except Exception:
                    cat_b = "?"
                bridge_lines.append(
                    f"- {pair.classification}: arxiv:{pair.paper_a} ({ta})"
                    f" ↔ arxiv:{pair.paper_b} ({tb}, {cat_b})"
                )
                if pair.reasoning_chain and pair.reasoning_chain != "[No reasoning captured]":
                    bridge_lines.append(f"  Reasoning: {pair.reasoning_chain[:200]}")
            bridge_text = "\n".join(bridge_lines)

            if all_direct:
                best = all_direct[0]
                pa = papers.get(best.paper_a, {})
                pb = papers.get(best.paper_b, {})
                try:
                    cats = json.loads(pb.get("categories", "[]") or "[]")
                    cat_b = cats[0] if cats else "?"
                except Exception:
                    cat_b = "?"
                prompt = (
                    f'Researcher asked: "{query}"\n\n'
                    f"Strongest cross-domain connection found:\n"
                    f"ML paper: \"{pa.get('title', best.paper_a)}\" (arxiv:{best.paper_a})\n"
                    f"  Abstract: {(pa.get('abstract') or '')[:300]}\n"
                    f"Outside-ML paper: \"{pb.get('title', best.paper_b)}\""
                    f" (arxiv:{best.paper_b}, {cat_b})\n"
                    f"  Abstract: {(pb.get('abstract') or '')[:300]}\n"
                    f"Connection: {best.reasoning_chain[:300]}\n\n"
                    f"Supporting papers:\n{papers_text}\n\n"
                    "Write exactly two short paragraphs:\n"
                    "Paragraph 1 (2 sentences): What this outside-ML paper says about the phenomenon "
                    "and why it matters to the researcher's question. Start with the field and paper.\n"
                    "Paragraph 2 (1-2 sentences): One specific experiment the researcher could run next week.\n"
                    "No labels, no jargon. Plain English."
                )
            else:
                prompt = (
                    f'You are analyzing papers found for this research query: "{query}"\n\n'
                    f"Top papers by relevance:\n{papers_text}\n\n"
                    f"Bridge detector findings:\n{bridge_text}\n\n"
                    "Your response MUST begin with 'Outside ML,' or 'Outside of ML,' if any "
                    "cross-domain connection is found. Answer exactly these three questions in "
                    "plain language, grounded only in what was actually found above. "
                    "Do not invent papers, results, or claims.\n\n"
                    "1. What do the most interesting papers show?\n"
                    "2. Is there any unexpected cross-domain connection in the findings?\n"
                    "3. What is one concrete, mathematically specific next step a researcher could pursue?\n\n"
                    "Keep each answer to 2-3 sentences. No labels or headers — just three paragraphs."
                )
        elif bridge_result.clusters:
            best = bridge_result.clusters[0]
            cats_str = " ↔ ".join(best.categories[:3]) if best.categories else "?"
            bridge_text = (
                f"Bridge clusters found ({cats_str}) but Gemma validation produced no confirmed pairs. "
                "Connections are based on embedding similarity only."
            )
            prompt = (
                f'You are analyzing papers found for this research query: "{query}"\n\n'
                f"Top papers by relevance:\n{papers_text}\n\n"
                f"Bridge detector findings:\n{bridge_text}\n\n"
                "Your response MUST begin with 'Outside ML,' or 'Outside of ML,' if any "
                "cross-domain connection is found. Answer exactly these three questions in "
                "plain language, grounded only in what was actually found above. "
                "Do not invent papers, results, or claims.\n\n"
                "1. What do the most interesting papers show?\n"
                "2. Is there any unexpected cross-domain connection in the findings?\n"
                "3. What is one concrete, mathematically specific next step a researcher could pursue?\n\n"
                "Keep each answer to 2-3 sentences. No labels or headers — just three paragraphs."
            )
        else:
            bridge_text = "No cross-domain bridges found — all papers appear to be from the same domain."
            prompt = (
                f'You are analyzing papers found for this research query: "{query}"\n\n'
                f"Top papers by relevance:\n{papers_text}\n\n"
                f"Bridge detector findings:\n{bridge_text}\n\n"
                "Your response MUST begin with 'Outside ML,' or 'Outside of ML,' if any "
                "cross-domain connection is found. Answer exactly these three questions in "
                "plain language, grounded only in what was actually found above. "
                "Do not invent papers, results, or claims.\n\n"
                "1. What do the most interesting papers show?\n"
                "2. Is there any unexpected cross-domain connection in the findings?\n"
                "3. What is one concrete, mathematically specific next step a researcher could pursue?\n\n"
                "Keep each answer to 2-3 sentences. No labels or headers — just three paragraphs."
            )

        llm_failed = False
        gemma_text = ""
        try:
            gemma_text = call_llm(prompt, timeout=30, temperature=0.3)
        except Exception:
            llm_failed = True

        if not gemma_text:
            _best = all_direct[0] if all_direct else (all_validated[0] if all_validated else None)
            if _best is not None:
                _pa = papers.get(_best.paper_a, {})
                _pb = papers.get(_best.paper_b, {})
                try:
                    _cats_b = json.loads(_pb.get("categories", "[]") or "[]")
                    _cat_b = _cats_b[0] if _cats_b else "?"
                except Exception:
                    _cat_b = "?"
                _n = len(all_direct) if all_direct else len(all_validated)
                _ta = _pa.get("title", _best.paper_a)
                _tb = _pb.get("title", _best.paper_b)
                _chain = (_best.reasoning_chain or "[none]")[:250]
                gemma_text = (
                    f"✓ {_n} cross-domain structural connection(s) found.\n"
                    "[Synthesis unavailable — network issue]\n\n"
                    "Strongest connection:\n"
                    f"{_tb} (arxiv:{_best.paper_b}, {_cat_b})\n"
                    f"↔ {_ta} (arxiv:{_best.paper_a})\n\n"
                    f"Structural correspondence: {_chain}\n\n"
                    "Open cross_domain_report.md for full analysis."
                )
            else:
                gemma_text = "[No cross-domain connections found. Full analysis in cross_domain_report.md.]"

    prefix_parts: list[str] = []
    if qrs_missing:
        prefix_parts.append(
            "Query relevance scoring unavailable (memory constraint). "
            "Showing cross-domain bridges found via structural analysis."
        )
    if bridge_result.structural_queries:
        sq_preview = " | ".join(bridge_result.structural_queries[:3])
        prefix_parts.append(f"Searched outside ML using: {sq_preview}")
    n_strong = len(strong_direct)
    n_related = len(related_direct)
    if (n_strong or n_related) and not llm_failed:
        n_frameworks = len({p.paper_b for p in all_direct})
        n_total = len(all_direct)
        prefix_parts.append(
            f"✓ {n_frameworks} outside-ML framework(s) found across {n_total} connection(s)."
        )
    if gemma_text and prefix_parts:
        gemma_text = "\n\n".join(prefix_parts) + "\n\n" + gemma_text

    console.print(
        Panel(
            gemma_text + f"\n\n[dim]Full analysis → ./{out_dir.name}/[/dim]",
            title="[bold]What I found[/bold]",
            border_style="cyan",
            padding=(0, 2),
        )
    )


def _run_search(query: str, console: Console, settings: dict) -> tuple | None:
    from .cli import execute_pipeline
    from .output_writer import save_session

    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
    if config.FAISS_PATH.exists():
        config.FAISS_PATH.unlink()

    from .db import init_db
    init_db()

    max_validate: Optional[int] = settings.get("max_validate")

    try:
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
    except Exception as exc:
        console.print(f"\n  [red]Pipeline error:[/red] {exc}\n  Try again or check your connection.")
        return None

    if result:
        bridge_result, papers = result
        out_dir, directions = save_session(query, bridge_result, papers)
        _gemma_synthesis_panel(query, bridge_result, papers, out_dir, console, directions=directions)
        return bridge_result, papers, out_dir
    return None


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
        f"  ·  Validate: [cyan]{settings.get('max_validate', 50)}[/cyan]"
        f"  ·  Clusters: [cyan]{settings.get('top_clusters', 3)}[/cyan]"
        f"  ·  [dim]/help for commands[/dim]\n"
    )


_FIELD_NAMES: dict[str, str] = {
    "math.DS": "Dynamical Systems",
    "cond-mat.stat-mech": "Statistical Physics",
    "nlin": "Nonlinear Dynamics",
    "nlin.CD": "Nonlinear Dynamics",
    "physics": "Physics",
    "physics.soc-ph": "Social Physics",
    "q-bio": "Quantitative Biology",
    "q-bio.NC": "Computational Neuroscience",
    "math": "Mathematics",
    "math.NA": "Numerical Analysis",
    "math.PR": "Probability Theory",
    "cs": "Computer Science",
    "cs.LG": "Machine Learning",
    "cs.AI": "Artificial Intelligence",
    "econ": "Economics",
    "eess": "Electrical Engineering",
    "stat": "Statistics",
    "cond-mat": "Condensed Matter",
}


def _cmd_help(console: Console) -> None:
    table = Table(title="Slash Commands", border_style="dim", show_header=True)
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description")
    table.add_row("/help",          "Show this message")
    table.add_row("/settings",      "Change LLM, embedding model, or pipeline parameters")
    table.add_row("/history",       "List past queries; re-run by number")
    table.add_row("/save [file]",   "Copy output files to a named location")
    table.add_row("/rerun",         "Re-run the most recent query")
    table.add_row("/fields",        "Show which arXiv categories were retrieved")
    table.add_row("/pairs",         "Show validated cross-domain pairs")
    table.add_row("/clear",         "Clear the screen and reprint session header")
    table.add_row("/quit",          "Exit")
    console.print(table)
    console.print("  [dim]Or just type your research problem and press Enter.[/dim]\n")


def _cmd_clear(console: Console, settings: dict) -> None:
    console.clear()
    _print_session_header(console, settings)


def _cmd_history(
    console: Console,
    history: list[str],
    run_fn: object,
) -> None:
    if not history:
        console.print("  [yellow]No queries yet.[/yellow]")
        return
    console.print()
    for i, q in enumerate(history, 1):
        console.print(f"  [cyan]{i}[/cyan]  {q}")
    try:
        choice = console.input("\n  Re-run query # (Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(history):
            run_fn(history[idx])  # type: ignore[operator]


def _cmd_save(console: Console, last_result: tuple | None, dest: str | None) -> None:
    if last_result is None:
        console.print("  [yellow]No results yet — run a query first.[/yellow]")
        return
    _, _, out_dir = last_result
    if not dest:
        try:
            dest = console.input("  Save to directory: ").strip()
        except (EOFError, KeyboardInterrupt):
            return
    if not dest:
        return
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    copied = 0
    for fname in ("cross_domain_report.md", "references.md", "connection_map.html"):
        src = out_dir / fname  # type: ignore[operator]
        if src.exists():
            shutil.copy2(src, dest_path / fname)
            copied += 1
    console.print(f"  [green]✓[/green] {copied} file(s) saved to [cyan]{dest}[/cyan]")


def _cmd_fields(console: Console, last_result: tuple | None) -> None:
    if last_result is None:
        console.print("  [yellow]No results yet — run a query first.[/yellow]")
        return
    _, papers, _ = last_result
    cat_info: dict[str, dict] = {}
    for p in papers.values():
        try:
            cats = json.loads(p.get("categories", "[]") or "[]")
            tag = p.get("query_tag", "") or ""
            channel = "structural" if tag.startswith("str") else "semantic"
        except Exception:
            continue
        if cats:
            cat = cats[0]
            if cat not in cat_info:
                cat_info[cat] = {"count": 0, "channel": channel}
            cat_info[cat]["count"] += 1

    table = Table(title="arXiv Fields Retrieved", border_style="dim")
    table.add_column("Category", style="cyan")
    table.add_column("Field Name")
    table.add_column("Papers", justify="right", style="green")
    table.add_column("Channel", style="dim")
    for cat, info in sorted(cat_info.items(), key=lambda x: -x[1]["count"]):
        prefix = cat.split(".")[0]
        name = _FIELD_NAMES.get(cat) or _FIELD_NAMES.get(prefix) or cat
        table.add_row(cat, name, str(info["count"]), info["channel"])
    console.print(table)


def _cmd_pairs(console: Console, last_result: tuple | None) -> None:
    if last_result is None:
        console.print("  [yellow]No results yet — run a query first.[/yellow]")
        return
    bridge_result, papers, _ = last_result
    pairs = getattr(bridge_result, "direct_cross_domain_pairs", None) or []
    if not pairs:
        console.print("  [yellow]No validated cross-domain pairs in last run.[/yellow]")
        return
    table = Table(title="Cross-Domain Pairs", border_style="dim")
    table.add_column("Type", style="green", no_wrap=True)
    table.add_column("Outside-ML Paper")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Reason")
    for pair in pairs:
        pb = papers.get(pair.paper_b, {})
        title = (pb.get("title") or pair.paper_b)[:60]
        try:
            cats = json.loads(pb.get("categories", "[]") or "[]")
            cat = cats[0] if cats else "?"
        except Exception:
            cat = "?"
        chain = (pair.reasoning_chain or "").replace("\n", " ")
        reason = chain.split(". ")[0][:80] if chain else "—"
        table.add_row(pair.classification, title, cat, reason)
    console.print(table)
