# Arxanon

Finds papers outside your field that describe the exact mathematical structure you're stuck on — from arXiv, with Gemma 4 validation, in plain English, with clickable links.

## What it does

You describe an ML phenomenon. Arxanon asks Gemma 4 to generate targeted arXiv search queries in ML vocabulary, then translates that same phenomenon into the native vocabulary of physics, mathematics, and related fields to search a second time. Both channels retrieve papers from arXiv, build a citation graph via Semantic Scholar, and embed everything for similarity search. Gemma 4 then validates which outside-ML papers share genuine structural correspondence with your ML papers — not just surface-level keyword overlap. The result is a `cross_domain_report.md` with synthesis and a `references.md` with direct arXiv links to the outside-ML papers, ranked by connection strength, plus an interactive `connection_map.html` visualization.

## Setup

**Python 3.10 or later required.**

```bash
git clone https://github.com/Serhii2009/arxanon
cd arxanon
pip install -e .
```

**For cloud inference (recommended):** get an [OpenRouter](https://openrouter.ai) API key and set it as an environment variable, or enter it in the setup wizard:

```bash
export OPENROUTER_API_KEY=sk-or-...
arxanon interactive
```

**For local inference:** install [Ollama](https://ollama.com), pull a Gemma 4 model, and run:

```bash
ollama pull gemma4:e2b
arxanon interactive
```

The first run launches a one-time setup wizard to choose your embedding model and LLM provider. Settings are saved to `~/.arxanon/settings.json`.

## Quickstart

```bash
arxanon interactive
```

Type a research problem at the prompt. Examples:

```
  Research problem > grokking in neural networks
  Research problem > learning rate warmup instability in transformers
  Research problem > emergent capabilities and scaling laws
```

## Example queries that work well

| Query | Outside-ML fields typically found |
|---|---|
| `grokking` | Nonlinear dynamics, statistical physics |
| `learning rate warmup instability` | Stiff ODEs, bifurcation theory |
| `emergent capabilities scaling laws` | Phase transitions, percolation theory |

## Slash commands

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/settings` | Change LLM provider, model, or pipeline parameters |
| `/history` | List past queries; re-run by number |
| `/save [path]` | Copy output files to a named directory |
| `/rerun` | Re-run the most recent query |
| `/fields` | Show which arXiv categories were retrieved |
| `/pairs` | Show validated cross-domain pairs in a table |
| `/clear` | Clear the screen |
| `/quit` | Exit |

## Output files

Each run creates a timestamped directory (e.g. `grokking_neural_networks/`) containing:

- **`cross_domain_report.md`** — synthesis from Gemma 4, cross-domain connections with reasoning, and the top retrieved papers
- **`references.md`** — outside-ML papers ranked by connection strength (strong first, then related), each with a direct arXiv link and one-sentence explanation
- **`connection_map.html`** — interactive D3.js network: click any node to open the paper on arxiv.org, hover any node to highlight its connections (everything else dims), STRUCTURAL edges visually dominant

## Scope

Works best for ML training dynamics phenomena — phenomena that occur *during* training and have a potential mathematical analogue in physics or applied mathematics. Results depend on Gemma 4 quality and the specificity of the query. A vague query finds vague connections; a precise phenomenon finds precise analogues.

Ollama with `gemma4:e2b` is significantly faster than cloud OpenRouter but may produce lower-quality reasoning chains.

## License

Apache 2.0 — see [LICENSE](LICENSE).
