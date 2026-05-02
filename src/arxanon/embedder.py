from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import faiss
import numpy as np

from . import config

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Unified embedding interface over sentence-transformers models.

    Handles model-specific requirements:
    - NV-Embed-v2: EOS token appended, instruction prefix for queries, batch_size=1
    - BAAI/bge-large-en-v1.5: optional query prefix, batch_size=32
    """

    def __init__(self, model_name: str = "") -> None:
        self.model_name = model_name or config.EMBED_MODEL
        self._model: SentenceTransformer | None = None

    @property
    def dim(self) -> int:
        if self._model is not None and hasattr(self._model, "get_sentence_embedding_dimension"):
            return self._model.get_sentence_embedding_dimension()
        return config.EMBED_DIMS.get(self.model_name, 768)

    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name, trust_remote_code=True)

        if "NV-Embed" in self.model_name:
            self._model.max_seq_length = 32768
            self._model.tokenizer.padding_side = "right"

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Encode texts into L2-normalised float32 embeddings.

        Args:
            texts: Strings to encode.
            is_query: If True, apply the query-specific instruction prefix.

        Returns:
            float32 array of shape (len(texts), dim), L2-normalised.
        """
        if self._model is None:
            self._load()

        if "NV-Embed" in self.model_name:
            return self._encode_nvembed(texts, is_query)
        return self._encode_bge(texts, is_query)

    def _encode_nvembed(self, texts: list[str], is_query: bool) -> np.ndarray:
        eos = self._model.tokenizer.eos_token
        texts = [t + eos for t in texts]
        kwargs: dict = dict(normalize_embeddings=True, batch_size=1, show_progress_bar=False)
        if is_query:
            kwargs["prompt"] = (
                "Instruct: Retrieve semantically similar scientific papers.\nQuery: "
            )
        embeddings = self._model.encode(texts, **kwargs)
        return np.array(embeddings, dtype=np.float32)

    def _encode_bge(self, texts: list[str], is_query: bool) -> np.ndarray:
        if is_query:
            prefix = "Represent this sentence for searching relevant passages: "
            texts = [prefix + t for t in texts]
        embeddings = self._model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return np.array(embeddings, dtype=np.float32)


def load_or_create_faiss_index(dim: int) -> faiss.IndexFlatIP:
    """Load FAISS index from disk, or create a new empty IndexFlatIP."""
    if config.FAISS_PATH.exists():
        return faiss.read_index(str(config.FAISS_PATH))
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return faiss.IndexFlatIP(dim)


def save_faiss_index(index: faiss.IndexFlatIP) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.FAISS_PATH))
