import sqlite3
from contextlib import contextmanager
from typing import Generator, Optional

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id   TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    abstract   TEXT NOT NULL,
    categories TEXT NOT NULL,
    date       TEXT NOT NULL,
    query_tag  TEXT NOT NULL DEFAULT '',
    authors    TEXT NOT NULL DEFAULT '[]',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS citation_edges (
    citing_arxiv_id TEXT NOT NULL,
    cited_arxiv_id  TEXT NOT NULL,
    edge_type       TEXT NOT NULL CHECK (edge_type IN ('direct', 'cocitation', 'bibcoupling')),
    PRIMARY KEY (citing_arxiv_id, cited_arxiv_id, edge_type)
);

CREATE TABLE IF NOT EXISTS embedding_index (
    arxiv_id  TEXT PRIMARY KEY,
    faiss_idx INTEGER NOT NULL UNIQUE
);
"""


def init_db() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _conn() as conn:
        conn.executescript(_SCHEMA)
    _migrate_db()


def _migrate_db() -> None:
    with _conn() as conn:
        try:
            conn.execute("ALTER TABLE papers ADD COLUMN authors TEXT NOT NULL DEFAULT '[]'")
        except Exception:
            pass  # column already exists


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_paper(
    arxiv_id: str,
    title: str,
    abstract: str,
    categories: str,
    date: str,
    query_tag: str = "",
    authors: str = "[]",
) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO papers (arxiv_id, title, abstract, categories, date, query_tag, authors)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(arxiv_id) DO UPDATE SET
                   title      = excluded.title,
                   abstract   = excluded.abstract,
                   categories = excluded.categories,
                   date       = excluded.date,
                   authors    = CASE WHEN excluded.authors != '[]'
                                     THEN excluded.authors
                                     ELSE papers.authors END,
                   query_tag  = CASE WHEN excluded.query_tag != ''
                                     THEN excluded.query_tag
                                     ELSE papers.query_tag END""",
            (arxiv_id, title, abstract, categories, date, query_tag, authors),
        )


def get_all_arxiv_ids() -> list[str]:
    with _conn() as conn:
        rows = conn.execute("SELECT arxiv_id FROM papers").fetchall()
    return [r["arxiv_id"] for r in rows]


def get_arxiv_ids_by_tag(tag: str) -> list[str]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT arxiv_id FROM papers WHERE query_tag = ?", (tag,)
        ).fetchall()
    return [r["arxiv_id"] for r in rows]


def get_papers_without_embeddings() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            """SELECT p.arxiv_id, p.abstract
               FROM papers p
               LEFT JOIN embedding_index ei ON p.arxiv_id = ei.arxiv_id
               WHERE ei.arxiv_id IS NULL"""
        ).fetchall()
    return [dict(r) for r in rows]


def upsert_citation_edge(citing: str, cited: str, edge_type: str) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO citation_edges
               (citing_arxiv_id, cited_arxiv_id, edge_type)
               VALUES (?, ?, ?)""",
            (citing, cited, edge_type),
        )


def store_embedding_idx(arxiv_id: str, faiss_idx: int) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO embedding_index (arxiv_id, faiss_idx) VALUES (?, ?)",
            (arxiv_id, faiss_idx),
        )


def get_embedding_idx_map() -> dict[str, int]:
    with _conn() as conn:
        rows = conn.execute("SELECT arxiv_id, faiss_idx FROM embedding_index").fetchall()
    return {r["arxiv_id"]: r["faiss_idx"] for r in rows}


def get_citation_edge_count() -> dict[str, int]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT edge_type, COUNT(*) AS cnt FROM citation_edges GROUP BY edge_type"
        ).fetchall()
    return {r["edge_type"]: r["cnt"] for r in rows}


def get_paper(arxiv_id: str) -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
    return dict(row) if row else None


def get_paper_count_by_tag() -> dict[str, int]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT query_tag, COUNT(*) AS cnt FROM papers GROUP BY query_tag"
        ).fetchall()
    return {r["query_tag"]: r["cnt"] for r in rows}


def get_unique_category_count() -> int:
    import json

    with _conn() as conn:
        rows = conn.execute("SELECT categories FROM papers").fetchall()
    cats: set[str] = set()
    for row in rows:
        try:
            cat_list = json.loads(row["categories"])
            if cat_list:
                cats.add(cat_list[0])
        except (json.JSONDecodeError, IndexError):
            pass
    return len(cats)


def compute_and_store_bibcoupling(coupling_threshold: int = 3) -> int:
    """Compute and store bibliographic coupling edges.

    Two papers have a bibcoupling edge when they share at least `coupling_threshold`
    common references (both have direct edges to the same cited papers).

    Idempotent: uses INSERT OR IGNORE, so repeated calls add only new edges.

    Returns:
        Number of NEW bibcoupling edges added by this call.
    """
    with _conn() as conn:
        before = conn.execute(
            "SELECT COUNT(*) FROM citation_edges WHERE edge_type = 'bibcoupling'"
        ).fetchone()[0]

        # Forward direction: a.citing < b.citing (lexicographic) to avoid duplicates
        conn.execute(
            """INSERT OR IGNORE INTO citation_edges (citing_arxiv_id, cited_arxiv_id, edge_type)
               SELECT a.citing_arxiv_id, b.citing_arxiv_id, 'bibcoupling'
               FROM citation_edges a
               JOIN citation_edges b ON a.cited_arxiv_id = b.cited_arxiv_id
               WHERE a.edge_type = 'direct'
                 AND b.edge_type = 'direct'
                 AND a.citing_arxiv_id < b.citing_arxiv_id
               GROUP BY a.citing_arxiv_id, b.citing_arxiv_id
               HAVING COUNT(*) >= ?""",
            (coupling_threshold,),
        )

        # Reverse direction: insert (b, a) for every (a, b) inserted above
        conn.execute(
            """INSERT OR IGNORE INTO citation_edges (citing_arxiv_id, cited_arxiv_id, edge_type)
               SELECT b.citing_arxiv_id, a.citing_arxiv_id, 'bibcoupling'
               FROM citation_edges a
               JOIN citation_edges b ON a.cited_arxiv_id = b.cited_arxiv_id
               WHERE a.edge_type = 'direct'
                 AND b.edge_type = 'direct'
                 AND a.citing_arxiv_id < b.citing_arxiv_id
               GROUP BY a.citing_arxiv_id, b.citing_arxiv_id
               HAVING COUNT(*) >= ?""",
            (coupling_threshold,),
        )

        after = conn.execute(
            "SELECT COUNT(*) FROM citation_edges WHERE edge_type = 'bibcoupling'"
        ).fetchone()[0]

        return after - before


def get_citation_pairs_for_nodes(arxiv_ids: list[str]) -> set[frozenset]:
    """Return all citation pairs (any edge_type) among the given paper set.

    Returns:
        Set of frozenset({id_a, id_b}) for every pair that has at least one
        citation relationship (direct, cocitation, or bibcoupling) in either direction.
    """
    if not arxiv_ids:
        return set()

    placeholders = ",".join("?" * len(arxiv_ids))
    with _conn() as conn:
        rows = conn.execute(
            f"""SELECT citing_arxiv_id, cited_arxiv_id
                FROM citation_edges
                WHERE citing_arxiv_id IN ({placeholders})
                  AND cited_arxiv_id IN ({placeholders})""",
            arxiv_ids + arxiv_ids,
        ).fetchall()

    return {frozenset({r["citing_arxiv_id"], r["cited_arxiv_id"]}) for r in rows}
