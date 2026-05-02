"""Phase 1 unit tests.

All tests use tmp_path fixtures to avoid touching ~/.arxanon.
The embedder tests mock model loading so no network download is required.
"""
import json
import sqlite3

import faiss
import numpy as np
import pytest

from arxanon import config
from arxanon.db import (
    get_arxiv_ids_by_tag,
    get_citation_edge_count,
    get_embedding_idx_map,
    get_paper,
    get_papers_without_embeddings,
    init_db,
    store_embedding_idx,
    upsert_citation_edge,
    upsert_paper,
)
from arxanon.embedder import Embedder
from arxanon.pipeline import find_top_similar_pairs

# ── DB tests ──────────────────────────────────────────────────────────────────

def test_db_init_creates_schema(tmp_path):
    init_db()

    conn = sqlite3.connect(str(tmp_path / "papers.db"))
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    conn.close()

    assert "papers" in tables
    assert "citation_edges" in tables
    assert "embedding_index" in tables


def test_upsert_paper_roundtrip():
    init_db()
    upsert_paper(
        arxiv_id="2206.01832",
        title="Test Paper",
        abstract="This is an abstract.",
        categories=json.dumps(["cs.LG", "stat.ML"]),
        date="2022-06-01",
        query_tag="semantic",
    )

    paper = get_paper("2206.01832")
    assert paper is not None
    assert paper["title"] == "Test Paper"
    assert paper["query_tag"] == "semantic"
    assert json.loads(paper["categories"]) == ["cs.LG", "stat.ML"]


def test_upsert_paper_does_not_overwrite_tag_with_empty():
    init_db()
    upsert_paper("test1", "T", "A", "[]", "2024-01-01", query_tag="semantic")
    upsert_paper("test1", "T updated", "A", "[]", "2024-01-01", query_tag="")

    paper = get_paper("test1")
    assert paper["title"] == "T updated"
    assert paper["query_tag"] == "semantic"


def test_upsert_citation_edge_types():
    init_db()
    upsert_paper("A", "Title A", "Abstract", "[]", "2024-01-01", "semantic")
    upsert_paper("B", "Title B", "Abstract", "[]", "2024-01-01", "structural")
    upsert_paper("C", "Title C", "Abstract", "[]", "2024-01-01", "semantic")

    upsert_citation_edge("A", "B", "direct")
    upsert_citation_edge("A", "C", "direct")
    upsert_citation_edge("B", "C", "cocitation")
    upsert_citation_edge("A", "B", "direct")  # duplicate — should be ignored

    counts = get_citation_edge_count()
    assert counts.get("direct", 0) == 2
    assert counts.get("cocitation", 0) == 1


def test_get_arxiv_ids_by_tag():
    init_db()
    upsert_paper("sem1", "S1", "A", "[]", "2024-01-01", "semantic")
    upsert_paper("sem2", "S2", "A", "[]", "2024-01-01", "semantic")
    upsert_paper("str1", "T1", "A", "[]", "2024-01-01", "structural")

    assert set(get_arxiv_ids_by_tag("semantic")) == {"sem1", "sem2"}
    assert get_arxiv_ids_by_tag("structural") == ["str1"]


def test_embedding_index_roundtrip():
    init_db()
    upsert_paper("p1", "P1", "A", "[]", "2024-01-01", "semantic")
    store_embedding_idx("p1", 0)

    idx_map = get_embedding_idx_map()
    assert idx_map == {"p1": 0}

    without = get_papers_without_embeddings()
    assert all(p["arxiv_id"] != "p1" for p in without)


# ── Embedder tests ────────────────────────────────────────────────────────────

class _MockSTModel:
    """Minimal SentenceTransformer mock that returns random normalised vectors."""

    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False, **kwargs):
        n = len(texts)
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((n, self._dim)).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / norms
        return vecs


def test_embedder_bge_encode_shape(monkeypatch):
    """Embedder.encode returns correctly shaped, normalised float32 array."""

    def _mock_load(self):
        self._model = _MockSTModel(dim=1024)

    monkeypatch.setattr(Embedder, "_load", _mock_load)

    embedder = Embedder("BAAI/bge-large-en-v1.5")
    texts = ["paper one about optimization", "paper two about dynamical systems", "paper three"]
    embeddings = embedder.encode(texts, is_query=False)

    assert embeddings.shape == (3, 1024)
    assert embeddings.dtype == np.float32
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_embedder_query_mode_prepends_prefix(monkeypatch):
    """In query mode, BGE embedder prepends the retrieval prefix."""
    captured: list[list[str]] = []

    class _CapturingModel(_MockSTModel):
        def encode(self, texts, **kwargs):
            captured.append(list(texts))
            return super().encode(texts, **kwargs)

    def _mock_load(self):
        self._model = _CapturingModel(dim=1024)

    monkeypatch.setattr(Embedder, "_load", _mock_load)

    embedder = Embedder("BAAI/bge-large-en-v1.5")
    embedder.encode(["hello world"], is_query=True)

    assert captured
    assert captured[0][0].startswith("Represent this sentence for searching relevant passages: ")


# ── Pipeline / FAISS tests ────────────────────────────────────────────────────

def _seed_db_and_faiss(tmp_path):
    """Insert 5 papers (3 semantic, 2 structural) and write a FAISS index.

    Designed so that (sem1, str1) are the most similar cross-domain pair.
    """
    init_db()

    papers = [
        ("sem1", "semantic", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        ("sem2", "semantic", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
        ("sem3", "semantic", np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)),
        ("str1", "structural", np.array([0.999, 0.045, 0.0, 0.0], dtype=np.float32)),
        ("str2", "structural", np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)),
    ]

    for arxiv_id, tag, _ in papers:
        upsert_paper(arxiv_id, f"Title {arxiv_id}", "Abstract text", "[]", "2024-01-01", tag)

    dim = 4
    index = faiss.IndexFlatIP(dim)
    vecs = np.stack([v for _, _, v in papers])
    faiss.normalize_L2(vecs)
    index.add(vecs)
    faiss.write_index(index, str(config.FAISS_PATH))

    for i, (arxiv_id, _, _) in enumerate(papers):
        store_embedding_idx(arxiv_id, i)


def test_find_top_pairs_cross_only(tmp_path):
    """Cross-only filter returns only semantic ↔ structural pairs."""
    _seed_db_and_faiss(tmp_path)

    pairs = find_top_similar_pairs(n=5, cross_only=True)
    assert len(pairs) > 0

    for id1, title1, cat1, id2, title2, cat2, score in pairs:
        from arxanon.db import get_paper as _get
        t1 = _get(id1)["query_tag"]
        t2 = _get(id2)["query_tag"]
        assert {t1, t2} == {"semantic", "structural"}, (
            f"Expected cross-domain pair, got {t1} ↔ {t2}"
        )


def test_find_top_pairs_top_match_correct(tmp_path):
    """The highest-scoring cross pair should be (sem1, str1)."""
    _seed_db_and_faiss(tmp_path)

    pairs = find_top_similar_pairs(n=1, cross_only=True)
    assert len(pairs) == 1

    id1, _, _, id2, _, _, score = pairs[0]
    assert {id1, id2} == {"sem1", "str1"}, f"Expected sem1/str1, got {id1}/{id2}"
    assert score > 0.99


def test_find_top_pairs_no_cross(tmp_path):
    """With cross_only=False, same-tag pairs are also returned."""
    _seed_db_and_faiss(tmp_path)

    pairs_cross = find_top_similar_pairs(n=10, cross_only=True)
    pairs_all = find_top_similar_pairs(n=10, cross_only=False)
    assert len(pairs_all) >= len(pairs_cross)
