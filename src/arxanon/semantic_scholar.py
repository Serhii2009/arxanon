import logging
import re
import time
from typing import Callable, Optional

import requests

from . import config
from .db import upsert_citation_edge

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1"


def _headers() -> dict[str, str]:
    h: dict[str, str] = {"User-Agent": "arxanon/0.1.0 (research tool)"}
    if config.S2_API_KEY:
        h["x-api-key"] = config.S2_API_KEY
    return h


def _get_json(url: str, params: Optional[dict] = None) -> Optional[dict]:
    """GET with exponential-backoff retry on rate limits."""
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=_headers(), timeout=30)
            if resp.status_code == 429:
                wait = 5 * (2**attempt)
                logger.warning("Rate limited by S2, waiting %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("S2 GET failed (%s): %s", url, exc)
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    return None


def _normalize_s2_arxiv_id(raw: str) -> str:
    return re.sub(r"v\d+$", "", raw)


def _batch_lookup(arxiv_ids: list[str]) -> dict[str, str]:
    """POST /paper/batch to map arXiv IDs -> S2 paper IDs.

    Processes in chunks of config.S2_BATCH_SIZE.
    Returns {arxiv_id: s2_paper_id}.
    """
    result: dict[str, str] = {}
    arxiv_set = set(arxiv_ids)

    for i in range(0, len(arxiv_ids), config.S2_BATCH_SIZE):
        chunk = arxiv_ids[i : i + config.S2_BATCH_SIZE]
        try:
            resp = requests.post(
                f"{_BASE_URL}/paper/batch",
                json={"ids": [f"arxiv:{aid}" for aid in chunk]},
                params={"fields": "paperId,externalIds"},
                headers=_headers(),
                timeout=30,
            )
            if resp.status_code == 429:
                time.sleep(10)
                continue
            resp.raise_for_status()
            papers = resp.json()
        except requests.RequestException as exc:
            logger.warning("S2 batch lookup failed: %s", exc)
            time.sleep(2)
            continue

        for paper in papers:
            if not paper:
                continue
            s2_id = paper.get("paperId")
            ext = paper.get("externalIds") or {}
            raw_ax = ext.get("ArXiv")
            if s2_id and raw_ax:
                ax_id = _normalize_s2_arxiv_id(raw_ax)
                if ax_id in arxiv_set:
                    result[ax_id] = s2_id

        time.sleep(config.S2_RATE_LIMIT_DELAY)

    return result


def fetch_and_store_citations(
    arxiv_ids: list[str],
    on_paper: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Fetch citation graph from Semantic Scholar and store in SQLite.

    For each paper, fetches its reference list and stores:
    - direct edges: this paper cites each reference in our DB
    - co-citation edges: pairs of references in our DB that are both cited by this paper

    Args:
        arxiv_ids: List of arXiv IDs already in the local DB.
        on_paper: Optional callback(completed, total) called after each paper is processed.

    Returns:
        Total number of edges stored across all types.
    """
    if not arxiv_ids:
        return 0

    arxiv_set = set(arxiv_ids)
    s2_map = _batch_lookup(arxiv_ids)

    total_edges = 0
    total = len(s2_map)

    for idx, (arxiv_id, s2_id) in enumerate(s2_map.items(), start=1):
        data = _get_json(
            f"{_BASE_URL}/paper/{s2_id}/references",
            params={"fields": "paperId,externalIds", "limit": 100},
        )
        time.sleep(config.S2_RATE_LIMIT_DELAY)

        if not data:
            if on_paper:
                on_paper(idx, total)
            continue

        db_refs: list[str] = []
        for ref in data.get("data", []):
            cited = ref.get("citedPaper") or {}
            ext = cited.get("externalIds") or {}
            raw_ax = ext.get("ArXiv")
            if not raw_ax:
                continue
            ref_id = _normalize_s2_arxiv_id(raw_ax)
            if ref_id in arxiv_set and ref_id != arxiv_id:
                db_refs.append(ref_id)

        for ref_id in db_refs:
            upsert_citation_edge(arxiv_id, ref_id, "direct")
            total_edges += 1

        for i in range(len(db_refs)):
            for j in range(i + 1, len(db_refs)):
                upsert_citation_edge(db_refs[i], db_refs[j], "cocitation")
                total_edges += 1

        if on_paper:
            on_paper(idx, total)

    return total_edges
