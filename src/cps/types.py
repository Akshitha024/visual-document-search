"""Core types: a Page = (page_id, image_or_synthetic_features). The benchmark
maps query -> page id via MaxSim over per-patch embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Page:
    page_id: str
    doc_id: str
    title: str | None = None
    # patch embeddings: (n_patches, embed_dim)


@dataclass
class PageEmbedding:
    page_id: str
    patches: NDArray[np.float64]  # (n_patches, d)


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    relevant_pages: tuple[str, ...]  # ground-truth page_ids


@dataclass(frozen=True)
class Hit:
    page_id: str
    score: float
    rank: int
