"""ColPali MaxSim scorer.

score(query, page) = sum_{i in query tokens} max_{j in page patches} <q_i, p_j>
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..types import Hit, PageEmbedding


def maxsim_one(query_toks: NDArray[np.float64], page: PageEmbedding) -> float:
    sim = query_toks @ page.patches.T  # (Q, P)
    return float(sim.max(axis=1).sum())


def maxsim_topk(
    query_toks: NDArray[np.float64], pages: list[PageEmbedding], top_k: int
) -> list[Hit]:
    scores = np.array([maxsim_one(query_toks, p) for p in pages], dtype=np.float64)
    top_k = min(top_k, len(pages))
    idxs = scores.argsort()[-top_k:][::-1]
    return [
        Hit(page_id=pages[int(i)].page_id, score=float(scores[i]), rank=rank + 1)
        for rank, i in enumerate(idxs)
    ]
