import json
import logging
import re
import time
from typing import Callable, Optional

import arxiv

from .db import upsert_paper

logger = logging.getLogger(__name__)

_INTER_QUERY_DELAY = 3.0          # seconds between consecutive queries
_BACKOFF_DELAYS = [90, 180, 300]  # wait times for successive 429 retries

_last_query_end: float = 0.0      # epoch time the last query finished


def _normalize_id(entry_id: str) -> str:
    """Extract canonical arXiv ID (no version suffix) from a full URL or bare ID.

    Examples:
        http://arxiv.org/abs/2206.01832v3 -> 2206.01832
        2206.01832v2                       -> 2206.01832
        math/0123456v1                     -> math/0123456
    """
    id_part = entry_id.split("/abs/")[-1]
    return re.sub(r"v\d+$", "", id_part)


def fetch_and_store_papers(
    query: str,
    max_results: int = 100,
    query_tag: str = "semantic",
    on_paper: Optional[Callable[[int], None]] = None,
) -> int:
    """Fetch papers from the arXiv API and store them in SQLite.

    Args:
        query: arXiv search query string.
        max_results: Maximum number of papers to retrieve.
        query_tag: Tag to associate with stored papers ('semantic' or 'structural').
        on_paper: Optional callback invoked with the cumulative count after each paper stored.

    Returns:
        Number of papers stored.
    """
    global _last_query_end
    if _last_query_end > 0:
        gap = time.time() - _last_query_end
        if gap < _INTER_QUERY_DELAY:
            time.sleep(_INTER_QUERY_DELAY - gap)

    logger.debug("arXiv query [%s]: %r", query_tag, query)
    client = arxiv.Client(
        page_size=min(100, max_results),
        delay_seconds=3.0,
        num_retries=3,
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    count = 0
    for attempt in range(len(_BACKOFF_DELAYS) + 1):
        try:
            for result in client.results(search):
                arxiv_id = _normalize_id(result.entry_id)
                author_names = [a.name for a in result.authors if a.name]
                upsert_paper(
                    arxiv_id=arxiv_id,
                    title=result.title.strip().replace("\n", " "),
                    abstract=result.summary.strip().replace("\n", " "),
                    categories=json.dumps(result.categories),
                    date=result.published.date().isoformat(),
                    query_tag=query_tag,
                    authors=json.dumps(author_names),
                )
                count += 1
                if on_paper:
                    on_paper(count)
            break
        except Exception as exc:
            status = getattr(exc, "status", None) or getattr(exc, "code", None)
            is_rate_limit = status == 429 or "429" in str(exc) or "rate" in str(exc).lower()
            if is_rate_limit and attempt < len(_BACKOFF_DELAYS):
                delay = _BACKOFF_DELAYS[attempt]
                logger.warning(
                    "arXiv is rate limiting this connection — waiting %ds before retry"
                    " (normal for heavy API use)", delay
                )
                time.sleep(delay)
            else:
                logger.warning(
                    "arXiv query [%s] failed: %s; using %d papers collected so far",
                    query_tag, exc, count,
                )
                break

    _last_query_end = time.time()
    return count
