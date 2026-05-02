# Arxanon

**Arxanon finds the papers you need but would never search for** — papers from fields you don't follow, written in vocabulary you don't use, describing the exact mathematical structure you're stuck on.

```
$ arxanon search "edge of stability gradient descent neural networks"

  ╔══════════════════════════════════════════════╗
  ║           A R X A N O N  v0.1.0              ║
  ║   Cross-Domain Structural Analogy Engine     ║
  ╚══════════════════════════════════════════════╝
  Embedding: BAAI/bge-large-en-v1.5

  ┌─ Retrieval ──────────────────────────────────────┐
  │  [semantic]    ████████████████████  100 papers  │
  │  [structural]  ████████████████████   87 papers  │
  │  Total: 187 papers · 9 arXiv categories          │
  └──────────────────────────────────────────────────┘

  ┌─ Top 5 Cross-Domain Pairs ───────────────────────────────────────────┐
  │  #   Score  Cat A       Title A                                       │
  │             Cat B       Title B                                       │
  │  ──────────────────────────────────────────────────────────────────  │
  │  1   0.784  [cs.LG]     Gradient Descent on Neural Networks...        │
  │             [math.DS]   Delayed Passage Through a Hopf Bifurcation... │
  └──────────────────────────────────────────────────────────────────────┘

  Top match: cs.LG ↔ math.DS (score 0.784)
  This suggests a structural connection between ML training dynamics
  and dynamical systems theory.
```

## What it does

Arxanon implements the dense-embedding generalization of [Swanson's Literature-Based Discovery](https://en.wikipedia.org/wiki/Literature-based_discovery) across all of arXiv. It treats **citation absence as a discovery signal**: papers that are semantically similar but have never cited each other are candidate cross-domain bridges.

**Phase 1 (current):** Retrieval and embedding validation — arXiv paper fetch, Semantic Scholar citation graph, NV-Embed-v2 embeddings, cross-domain similarity search.

**Phase 2 (coming):** Citation exclusion filter + HDBSCAN bridge clustering.

**Phase 3 (coming):** Gemma 4 structural analogy verification via Ollama.

**Phase 4 (coming):** Bridge reports, translation dictionaries, experiment sketches, D3.js bridge map.

## Installation

```bash
# From source (until PyPI release)
git clone https://github.com/Serhii2009/arxanon
cd arxanon
pip install -e .
```

For GPU support with NV-Embed-v2 (requires 16GB+ VRAM), install `faiss-gpu` instead of `faiss-cpu`:

```bash
pip install -e . && pip install faiss-gpu
```

## Quickstart

```bash
# Use BAAI/bge-large-en-v1.5 as a fast CPU-friendly model (recommended for testing)
ARXANON_EMBED_MODEL="BAAI/bge-large-en-v1.5" arxanon search "edge of stability gradient descent"

# With a custom structural companion query
arxanon search "edge of stability gradient descent" \
  --structural-query "delayed bifurcation dynamical systems slow passage unstable equilibrium"

# Production (NV-Embed-v2, requires GPU)
arxanon search "edge of stability gradient descent"

# Get help
arxanon --help
arxanon search --help
```

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `ARXANON_EMBED_MODEL` | `nvidia/NV-Embed-v2` | Embedding model. Use `BAAI/bge-large-en-v1.5` for CPU. |
| `ARXANON_DATA_DIR` | `~/.arxanon` | Where SQLite DB and FAISS index are stored. |
| `S2_API_KEY` | _(none)_ | Semantic Scholar API key (improves rate limits). |

## Architecture

The system is designed for progressive assembly across four phases:

```
arXiv API ──► SQLite papers DB ──► FAISS embedding index
                  │                        │
Semantic Scholar  ▼                        ▼
citation graph ──► citation_edges ──► cross-domain pairs
                                           │
                                    (Phase 2: bridge detection)
                                    (Phase 3: Gemma 4 validation)
                                    (Phase 4: bridge reports)
```

See [ARXANON_DESIGN_DOCUMENT.md](ARXANON_DESIGN_DOCUMENT.md) for the full architecture.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
