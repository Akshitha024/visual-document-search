"""Retrieval metrics tailored to multi-page documents.

Slightly different from a standard IR set:
  - recall@k     : fraction of relevant pages in top-k
  - mrr@k        : 1 / rank of first relevant page
  - page_hit_at_1 / 5 / 10
  - doc_hit_at_5 : does at least one page of the right doc appear in top-5
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..types import Hit


@dataclass
class QResult:
    qid: str
    hits: list[Hit]
    relevant_pages: set[str]
    relevant_docs: set[str]


def page_hit_at_k(r: QResult, k: int) -> float:
    hit = {h.page_id for h in r.hits[:k]} & r.relevant_pages
    return 1.0 if hit else 0.0


def doc_hit_at_k(r: QResult, k: int, page_to_doc: dict[str, str]) -> float:
    hit_docs = {page_to_doc.get(h.page_id, "") for h in r.hits[:k]} & r.relevant_docs
    return 1.0 if hit_docs else 0.0


def recall_at_k(r: QResult, k: int) -> float:
    if not r.relevant_pages:
        return 0.0
    hit = {h.page_id for h in r.hits[:k]} & r.relevant_pages
    return len(hit) / len(r.relevant_pages)


def mrr_at_k(r: QResult, k: int) -> float:
    for i, h in enumerate(r.hits[:k], start=1):
        if h.page_id in r.relevant_pages:
            return 1.0 / i
    return 0.0


def aggregate(
    results: Sequence[QResult], ks: Sequence[int], page_to_doc: dict[str, str]
) -> dict[str, float]:
    out: dict[str, float] = {}
    n = max(1, len(results))
    for k in ks:
        out[f"page_hit@{k}"] = sum(page_hit_at_k(r, k) for r in results) / n
        out[f"doc_hit@{k}"] = sum(doc_hit_at_k(r, k, page_to_doc) for r in results) / n
        out[f"recall@{k}"] = sum(recall_at_k(r, k) for r in results) / n
        out[f"mrr@{k}"] = sum(mrr_at_k(r, k) for r in results) / n
    return out
