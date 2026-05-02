"""Phase 2, Layer 2: persistent homology via giotto-tda (optional).

Detects topological gaps — regions of conceptual space surrounded by papers
from different domains but with no paper in the centre.  H₁ persistent cycles
correspond to such gaps.

The giotto-tda import is guarded inside compute_tda() so the rest of the
package loads even when giotto-tda is not installed.  All callers must handle
TDAResult(enabled=False) gracefully.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

from . import config
from .clusters import PersistentCycle, TDAResult

logger = logging.getLogger(__name__)

_MIN_PAPERS_FOR_TDA = 4


def compute_tda(
    arxiv_ids: list[str],
    vectors: np.ndarray,
    paper_categories: dict[str, list[str]],
    percentile_threshold: float = config.TDA_PERSISTENCE_PERCENTILE,
) -> TDAResult:
    """Compute H₁ persistent homology on the bridge-paper embedding space.

    Args:
        arxiv_ids: Paper IDs corresponding to rows of *vectors*.
        vectors: (n, dim) L2-normalised float32 vectors for bridge papers only.
        paper_categories: Mapping of arxiv_id → list of arXiv category strings.
        percentile_threshold: Retain only cycles with persistence >= this
                              percentile of all observed persistence values.

    Returns:
        TDAResult with enabled=False when giotto-tda is not installed or when
        there are too few papers.  All other modules must tolerate this.
    """
    if len(arxiv_ids) < _MIN_PAPERS_FOR_TDA:
        return TDAResult(
            enabled=False,
            n_cycles=0,
            cycles=[],
            warning=f"Too few bridge papers ({len(arxiv_ids)}) for TDA (need >= {_MIN_PAPERS_FOR_TDA})",
        )

    try:
        from gtda.homology import VietorisRipsPersistence  # type: ignore[import]
    except ImportError:
        warnings.warn(
            "giotto-tda not installed — topological gap detection disabled.  "
            "Install with: pip install 'arxanon[tda]'",
            ImportWarning,
            stacklevel=2,
        )
        return TDAResult(
            enabled=False,
            n_cycles=0,
            cycles=[],
            warning="giotto-tda not installed",
        )

    # Cosine distance matrix (L2-normalised → dot product = cosine similarity)
    cos_sim = vectors @ vectors.T
    dist_matrix = np.clip(1.0 - cos_sim, 0.0, 2.0).astype(np.float32)
    np.fill_diagonal(dist_matrix, 0.0)

    try:
        vr = VietorisRipsPersistence(
            metric="precomputed",
            homology_dimensions=[1],
            n_jobs=-1,
        )
        diagrams = vr.fit_transform([dist_matrix])
    except Exception as exc:
        logger.warning("TDA computation failed: %s", exc)
        return TDAResult(enabled=False, n_cycles=0, cycles=[], warning=str(exc))

    # diagrams[0] shape: (n_points, 3) — columns: birth, death, dimension
    diagram = diagrams[0]
    h1_mask = diagram[:, 2] == 1  # H₁ only
    h1 = diagram[h1_mask]

    if len(h1) == 0:
        return TDAResult(enabled=True, n_cycles=0, cycles=[])

    persistence = h1[:, 1] - h1[:, 0]  # death - birth
    threshold = float(np.percentile(persistence, percentile_threshold)) if len(persistence) > 1 else 0.0

    persistent_mask = persistence >= threshold
    persistent_h1 = h1[persistent_mask]
    persistent_pers = persistence[persistent_mask]

    cycles: list[PersistentCycle] = []
    for idx in range(len(persistent_h1)):
        birth = float(persistent_h1[idx, 0])
        death = float(persistent_h1[idx, 1])
        pers = float(persistent_pers[idx])

        # Boundary papers: papers whose closest neighbour distance is near birth radius.
        # This is an approximation — the actual cycle representative requires ripser.
        radius = birth + 0.05
        boundary_indices = [
            i for i in range(len(arxiv_ids))
            if any(
                dist_matrix[i, j] <= radius
                for j in range(len(arxiv_ids))
                if j != i
            )
        ]
        boundary_papers = [arxiv_ids[i] for i in boundary_indices[:20]]

        # Distinct top-level categories of boundary papers
        cats = sorted({
            c.split(".")[0]
            for pid in boundary_papers
            if pid in paper_categories
            for c in paper_categories[pid][:1]
        })

        cycles.append(PersistentCycle(
            cycle_id=idx,
            birth=birth,
            death=death,
            persistence=pers,
            boundary_papers=boundary_papers,
            categories=cats,
        ))

    return TDAResult(enabled=True, n_cycles=len(cycles), cycles=cycles)
