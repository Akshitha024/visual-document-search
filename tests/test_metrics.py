from __future__ import annotations

import math

from cps.scoring.metrics import (
    QResult,
    doc_hit_at_k,
    mrr_at_k,
    page_hit_at_k,
    recall_at_k,
)
from cps.types import Hit


def _hits(ids: list[str]) -> list[Hit]:
    return [Hit(page_id=d, score=1.0 - i * 0.1, rank=i + 1) for i, d in enumerate(ids)]


def test_page_hit_at_k_finds() -> None:
    r = QResult(qid="q", hits=_hits(["a", "b", "c"]), relevant_pages={"b"}, relevant_docs=set())
    assert page_hit_at_k(r, 3) == 1.0
    assert page_hit_at_k(r, 1) == 0.0


def test_doc_hit_at_k_via_mapping() -> None:
    r = QResult(qid="q", hits=_hits(["a", "b"]), relevant_pages={"x"}, relevant_docs={"D1"})
    pt = {"a": "D2", "b": "D1"}
    assert doc_hit_at_k(r, 2, pt) == 1.0
    assert doc_hit_at_k(r, 1, pt) == 0.0


def test_recall_at_k() -> None:
    r = QResult(
        qid="q", hits=_hits(["a", "b", "c"]), relevant_pages={"a", "c", "d"}, relevant_docs=set()
    )
    assert math.isclose(recall_at_k(r, 5), 2 / 3)


def test_mrr_at_k() -> None:
    r = QResult(qid="q", hits=_hits(["x", "y", "z"]), relevant_pages={"y"}, relevant_docs=set())
    assert mrr_at_k(r, 5) == 0.5
