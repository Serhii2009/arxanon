<div align="center">

# ARXANON

**Cross-domain research engine for AI/ML**

[![PyPI version](https://img.shields.io/pypi/v/arxanon?cachebust=0)](https://pypi.org/project/arxanon/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/arxanon/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

</div>

---

Arxanon takes a description of an ML phenomenon, generates arXiv search queries in the native vocabulary of adjacent scientific fields, and uses an LLM to determine which cross-domain paper pairs share the same underlying mathematical structure. Each run produces a full analysis report, a ranked reference list, and an interactive connection graph — grounded entirely in papers retrieved from arXiv in that session.

## The problem it solves

A researcher encountering grokking knows the ML vocabulary: delayed generalization, phase transition in loss, weight norm growth. The relevant physics literature uses none of those terms. The same phenomenon appears in statistical physics as _critical slowing down_ and in the theory of dynamical systems as _rate-dependent bifurcation delay_. Standard arXiv search, Google Scholar, and semantic similarity over ML corpora will not surface those papers, because the connection has never been named explicitly in an ML paper. Arxanon finds them by translating the phenomenon into the vocabulary of each candidate field before searching.

## How it works

1. Gemma 4 generates ML search queries from the researcher's question, targeting papers that study the phenomenon during training.
2. Gemma 4 translates the phenomenon into structural vocabulary from adjacent fields — dynamical systems, statistical physics, neuroscience, control theory, economics — and generates queries using the terminology of those fields.
3. arXiv is searched with both sets of queries; the paper budget is split evenly between the two channels.
4. Papers are embedded with bge-large-en-v1.5 and indexed in FAISS; a citation graph is built via Semantic Scholar to identify which similar-looking paper pairs are already in the same citation network.
5. Gemma 4 validates whether cross-domain paper pairs (semantically similar but not citation-connected) share the same mathematical object or mechanism, classifying each as STRUCTURAL, METHODOLOGICAL, THEMATIC, or discarded.
6. Results are synthesized into a report with grounded claims and a suggested experiment derived from the strongest outside-ML finding.

Asking an LLM to suggest relevant physics papers directly will only surface papers it has already connected to ML during training. The vocabulary translation step finds papers where the connection has not yet been made in the ML literature.

## Installation

```bash
pip install arxanon
```

To use OpenRouter (recommended):

```bash
export OPENROUTER_API_KEY=your_key_here
```

Ollama works as a free local alternative with no API key required, but Ollama must be running with a Gemma model installed:

```bash
ollama serve
ollama pull gemma4:e2b
```

## Usage

```
arxanon
```

On first run, a one-time setup wizard configures your embedding model, LLM provider, and pipeline parameters. Settings are saved to `~/.arxanon/settings.json`.

```
╭──────────────────────────────────────────────────────────────────────╮
│  A R X A N O N  v1.0.0                                               │
│  Cross-domain research engine for AI/ML                              │
╰──────────────────────────────────────────────────────────────────────╯
  Embedding: bge-large-en-v1.5  ·  LLM: OpenRouter · google/gemma-4-31b-it
  Results: 100  ·  Validate: 50  ·  Clusters: 3  ·  /help for commands


  Research problem > grokking delayed generalization in neural networks

Generating search queries…

  ML search                                        Structural vocabulary
  "grokking delayed generalization training"       "critical slowing down bifurcation"
  "phase transition generalization dynamics"       "rate-dependent tipping nonlinear systems"
  "weight norm growth algorithmic alignment"       "slow manifold emergence complex systems"

Fetching papers from arXiv…
✓ 186 papers — 93 ML · 93 structural

Building citation network…
✓ Citation graph — 408 direct · 1 187 co-citation edges

Computing embeddings…
✓ 186 papers indexed

[Bridge detection → TDA → HDBSCAN clustering]
✓ 3 bridge clusters found

Validating cross-domain pairs with Gemma…
✓ Validation complete — 12 pairs

Running direct cross-domain comparison…
✓ 5 cross-domain pair(s)

╭─ What I found ────────────────────────────────────────────────────────────────╮
│  Searched outside ML using: critical slowing down bifurcation |              │
│  rate-dependent tipping nonlinear systems | slow manifold emergence          │
│  ✓ 3 outside-ML framework(s) found across 5 connection(s).                  │
│                                                                              │
│  **What we found:** A paper in nonlinear dynamics on rate-dependent          │
│  bifurcation delay [GROUNDED: arxiv:XXXX] shows the pre-transition          │
│  duration scales with a specific exponent of the rate parameter —            │
│  matching the relationship between grokking onset and learning rate          │
│  schedule length. [INFERRED] This predicts a quantitative relationship       │
│  not yet measured in the ML literature.                                      │
│                                                                              │
│  **Experiment:** Rate-dependent bifurcation delay predicts that halving      │
│  the learning rate schedule length shifts the generalization transition      │
│  step by a specific factor. Test: train the same network at 5 schedule      │
│  lengths and measure the step at which test accuracy first exceeds train     │
│  accuracy by < 1%.                                                           │
│                                                                              │
│  Full analysis → ./grokking_delayed_generalization/                          │
╰──────────────────────────────────────────────────────────────────────────────╯
```

> Paper counts, query text, and arXiv IDs in this example are illustrative. Actual output depends on arXiv's current index and what the LLM generates for your query.

### Slash commands

| Command        | Description                                    |
| -------------- | ---------------------------------------------- |
| `/help`        | Show all commands                              |
| `/settings`    | Change provider, model, or pipeline parameters |
| `/history`     | List past queries — select to re-run           |
| `/fields`      | Show which arXiv categories were retrieved     |
| `/pairs`       | Show validated cross-domain pairs in a table   |
| `/save [dest]` | Copy output files to a named location          |
| `/rerun`       | Re-run the most recent query                   |
| `/quit`        | Exit                                           |

## Output files

Each run writes a folder named after the first ML query generated (e.g. `grokking_delayed_generalization/`). Inside:

| File                     | Contents                                                                                                                                                                                                                                                                               |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cross_domain_report.md` | Full analysis: synthesis with inline provenance tags (`[GROUNDED: arxiv:X]`, `[INFERRED]`, `[SPECULATIVE]`), cross-domain connections grouped by outside-ML paper with reasoning, top ML papers by relevance, and a suggested experiment derived from the strongest outside-ML finding |
| `references.md`          | Outside-ML papers found in the run, deduplicated and ranked by connection strength (STRUCTURAL first, then METHODOLOGICAL and THEMATIC), with clickable arXiv links                                                                                                                    |
| `connection_map.html`    | Interactive D3.js force graph of the paper network — hover any node to highlight its connections, click to open the arXiv page                                                                                                                                                         |

## Configuration

Type `/settings` at the prompt to change your provider, model, or pipeline parameters:

| Parameter            | Default               | Range  | Description                               |
| -------------------- | --------------------- | ------ | ----------------------------------------- |
| LLM provider / model | Ollama · `gemma4:e2b` | —      | OpenRouter (cloud) or Ollama (local)      |
| `max_results`        | 100                   | 10–500 | Papers fetched per query                  |
| `max_validate`       | 50                    | 1–100  | Bridge pairs sent to Gemma for validation |

All settings persist between sessions.

## Scope and limitations

The tool works well when:

- The ML phenomenon has a vocabulary gap to physics or mathematics (grokking, loss spikes, learning rate warmup, emergent capabilities during training).
- The relevant outside literature exists on arXiv.

The tool adds less value when:

- The cross-domain name is already established in the ML literature. Reward hacking is already called Goodhart's Law in ML papers; the structural channel retrieves nothing that standard search would miss.
- The relevant literature is primarily in non-arXiv journals — behavioral economics, evolutionary biology, clinical neuroscience, and most social science are largely absent from arXiv.

## Technical notes

- Embedding model: `BAAI/bge-large-en-v1.5` (1024-dim, CPU). `nvidia/NV-Embed-v2` (4096-dim, GPU) is available via the setup wizard.
- LLM: Gemma 4 31B via OpenRouter (`google/gemma-4-31b-it`, recommended) or any local model via Ollama.
- Vector index: FAISS; citation graph via Semantic Scholar; topological structure detection via persistent homology (`pip install "arxanon[tda]"`).

---

**Built by [Serhii Kravchenko](https://www.linkedin.com/in/serhii-kravchenko1/) 🔥**
