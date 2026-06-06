"""Drive a synthetic ColPali-style retrieval benchmark end to end."""

from __future__ import annotations

import json
import random
from pathlib import Path

from loguru import logger

from .encoders.colpali_synth import encode_page, encode_query
from .scoring.maxsim import maxsim_topk
from .scoring.metrics import QResult, aggregate

# Synthetic dataset: 5 documents, 4 pages each, each page has a topic;
# queries target a topic plus a specific document.
_TOPICS = ["climate", "vaccines", "machine learning", "monetary policy", "antibiotics"]
_DOCS_PER_TOPIC = 5
_PAGES_PER_DOC = 4


def build_corpus() -> tuple[
    list[tuple[str, str, str]], list[tuple[str, str, tuple[str, ...], str]], dict[str, str]
]:
    pages = []
    page_to_doc = {}
    for ti, topic in enumerate(_TOPICS):
        for di in range(_DOCS_PER_TOPIC):
            doc_id = f"doc_{ti}_{di}"
            for pi in range(_PAGES_PER_DOC):
                page_id = f"{doc_id}__p{pi}"
                pages.append((page_id, doc_id, topic))
                page_to_doc[page_id] = doc_id
    queries = []
    rng = random.Random(7)
    for ti, topic in enumerate(_TOPICS):
        for di in range(_DOCS_PER_TOPIC):
            qid = f"q_{ti}_{di}"
            text = f"explain the {topic} in document {di}"
            # relevant pages are all pages of that specific (topic, doc) pair
            rel = tuple(f"doc_{ti}_{di}__p{pi}" for pi in range(_PAGES_PER_DOC))
            queries.append((qid, text, rel, topic))
            rng.random()  # bump rng for stable ordering
    return pages, queries, page_to_doc


def bench(out_dir: Path, top_k: int = 10) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_meta, queries_meta, page_to_doc = build_corpus()
    embeddings = [encode_page(pid, topic) for (pid, _, topic) in pages_meta]
    per_query_results: list[QResult] = []
    per_query_records: list[dict[str, object]] = []
    for qid, text, rel, _ in queries_meta:
        q_toks = encode_query(text)
        hits = maxsim_topk(q_toks, embeddings, top_k)
        relevant_docs = {page_to_doc[r] for r in rel if r in page_to_doc}
        qr = QResult(qid=qid, hits=hits, relevant_pages=set(rel), relevant_docs=relevant_docs)
        per_query_results.append(qr)
        per_query_records.append(
            {
                "qid": qid,
                "text": text,
                "hits": [{"page_id": h.page_id, "rank": h.rank, "score": h.score} for h in hits],
                "relevant_pages": list(rel),
            }
        )

    ks = [1, 3, 5, 10]
    metrics = aggregate(per_query_results, ks, page_to_doc)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out_dir / "queries.jsonl").open("w") as f:
        for r in per_query_records:
            f.write(json.dumps(r) + "\n")

    # also save the (page, doc, topic) layout so plots can color by topic
    layout = {
        "pages": [
            {"page_id": pid, "doc_id": did, "topic": topic} for (pid, did, topic) in pages_meta
        ]
    }
    (out_dir / "layout.json").write_text(json.dumps(layout))
    logger.info("wrote metrics + queries + layout to {}", out_dir)
    return metrics
