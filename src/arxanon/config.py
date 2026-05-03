import os
from pathlib import Path

EMBED_MODEL = os.getenv("ARXANON_EMBED_MODEL", "BAAI/bge-large-en-v1.5")

EMBED_DIMS: dict[str, int] = {
    "nvidia/NV-Embed-v2": 4096,
    "BAAI/bge-large-en-v1.5": 1024,
}

DATA_DIR = Path(os.getenv("ARXANON_DATA_DIR", Path.home() / ".arxanon"))
DB_PATH = DATA_DIR / "papers.db"
FAISS_PATH = DATA_DIR / "embeddings.faiss"

S2_API_KEY = os.getenv("S2_API_KEY", "")
S2_BATCH_SIZE = 100
S2_RATE_LIMIT_DELAY = 1.1

SIMILARITY_THRESHOLD = 0.72

# ── Phase 2: bridge detection ─────────────────────────────────────────────────
COUPLING_THRESHOLD: int = int(os.getenv("ARXANON_COUPLING_THRESHOLD", "3"))
TDA_ENABLED: bool = os.getenv("ARXANON_TDA_ENABLED", "1") != "0"
TDA_PERSISTENCE_PERCENTILE: float = 75.0
HDBSCAN_MIN_CLUSTER_SIZE: int = int(os.getenv("ARXANON_HDBSCAN_MIN_SIZE", "3"))
HDBSCAN_MIN_SAMPLES: int = 2

# Bridge score composite weights (must sum to 1.0)
BRIDGE_WEIGHT_DOMAIN: float = 0.24
BRIDGE_WEIGHT_COHERENCE: float = 0.20
BRIDGE_WEIGHT_ISOLATION: float = 0.24
BRIDGE_WEIGHT_TOPOLOGY: float = 0.12
BRIDGE_WEIGHT_QUERY: float = 0.20

# ── Phase 3: Gemma 4 via Ollama ───────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("ARXANON_OLLAMA_URL", "http://localhost:11434")
GEMMA_MODEL: str = os.getenv("ARXANON_GEMMA_MODEL", "gemma4:e2b")
MAX_BRIDGE_VALIDATIONS: int = int(os.getenv("ARXANON_MAX_VALIDATIONS", "50"))
# Simple mode: single plain-text call instead of multi-step function calling.
# Default ON — works with small models (e2b/2b) on CPU. Set to "0" for function-calling mode.
GEMMA_SIMPLE_MODE: bool = os.getenv("ARXANON_GEMMA_SIMPLE", "1") != "0"

# ── OpenRouter (optional cloud LLM backend) ───────────────────────────────────
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
USE_OPENROUTER: bool = bool(OPENROUTER_API_KEY)
