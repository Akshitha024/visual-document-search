"""Five charts for the ColPali-style benchmark.

- metric @ k curves (recall, mrr, page_hit, doc_hit overlaid)
- per-query top-rank histogram (where does the first relevant page land?)
- per-topic accuracy bars (page_hit@5 grouped by topic)
- score-margin scatter (rank-1 minus rank-2 score per query)
- similarity matrix of query embeddings (cosine between queries)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _load_json(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {}
    out: dict[str, Any] = json.loads(p.read_text())
    return out


def _load_jsonl(p: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not p.exists():
        return out
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def plot_metric_curves(metrics_path: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    m = _load_json(metrics_path)
    if not m:
        out.write_bytes(b"")
        return out
    ks = sorted({int(k.split("@")[1]) for k in m if "@" in k})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for series in ("page_hit", "doc_hit", "recall", "mrr"):
        ys = [m.get(f"{series}@{k}", 0) for k in ks]
        ax.plot(ks, ys, marker="o", linewidth=2, label=series)
    ax.set_xticks(ks)
    ax.set_xlabel("k")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title("ColPali-style retrieval metrics @ k")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_first_relevant_rank(queries_path: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_jsonl(queries_path)
    if not rows:
        out.write_bytes(b"")
        return out
    ranks: list[int] = []
    for r in rows:
        rel = set(r.get("relevant_pages") or [])
        first = None
        for h in r["hits"]:
            if h["page_id"] in rel:
                first = h["rank"]
                break
        ranks.append(first if first is not None else 999)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    in_range = [r for r in ranks if r <= 20]
    miss = sum(1 for r in ranks if r > 20)
    ax.hist(in_range, bins=range(1, 22), color="#1f77b4", edgecolor="black", alpha=0.7)
    ax.set_xticks(list(range(1, 21, 2)))
    ax.set_xlabel("rank of first relevant page")
    ax.set_ylabel("queries")
    ax.set_title(f"Where the first relevant page lands ({miss} queries miss top-20)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_per_topic_accuracy(queries_path: Path, layout_path: Path, out: Path, k: int = 5) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_jsonl(queries_path)
    layout = _load_json(layout_path)
    if not rows or not layout:
        out.write_bytes(b"")
        return out
    page_topic: dict[str, str] = {p["page_id"]: p["topic"] for p in layout["pages"]}
    by_topic_hits: dict[str, int] = defaultdict(int)
    by_topic_total: dict[str, int] = defaultdict(int)
    for r in rows:
        rel = set(r.get("relevant_pages") or [])
        if not rel:
            continue
        # topic = the topic of any relevant page
        topic = next((page_topic[p] for p in rel if p in page_topic), "?")
        by_topic_total[topic] += 1
        top_ids = {h["page_id"] for h in r["hits"][:k]}
        if rel & top_ids:
            by_topic_hits[topic] += 1
    topics = sorted(by_topic_total)
    rates = [by_topic_hits[t] / by_topic_total[t] for t in topics]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(topics)), 4.5))
    bars = ax.bar(topics, rates, color="#2ca02c")
    for bar, v in zip(bars, rates, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{v:.2f}",
            ha="center",
            fontsize=9,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"page_hit@{k}")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    ax.set_title(f"Per-topic page_hit@{k}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_score_margin(queries_path: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_jsonl(queries_path)
    if not rows:
        out.write_bytes(b"")
        return out
    margins = []
    correct = []
    for r in rows:
        hits = r["hits"]
        if len(hits) < 2:
            continue
        rel = set(r.get("relevant_pages") or [])
        margins.append(hits[0]["score"] - hits[1]["score"])
        correct.append(1 if hits[0]["page_id"] in rel else 0)
    if not margins:
        out.write_bytes(b"")
        return out
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#2ca02c" if c else "#d62728" for c in correct]
    ax.scatter(range(len(margins)), margins, c=colors, alpha=0.7, s=70, edgecolor="black")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("query index")
    ax.set_ylabel("rank-1 score minus rank-2 score")
    correct_n = sum(correct)
    ax.set_title(f"Score margin per query (green = top-1 correct, n={correct_n}/{len(margins)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_topic_confusion(queries_path: Path, layout_path: Path, out: Path) -> Path:
    """Heatmap: rows = query topic, cols = top-1 retrieved-page topic. Diagonal
    = correct topic.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_jsonl(queries_path)
    layout = _load_json(layout_path)
    if not rows or not layout:
        out.write_bytes(b"")
        return out
    page_topic: dict[str, str] = {p["page_id"]: p["topic"] for p in layout["pages"]}
    topics = sorted({p["topic"] for p in layout["pages"]})
    idx = {t: i for i, t in enumerate(topics)}
    mat = np.zeros((len(topics), len(topics)), dtype=np.float64)
    for r in rows:
        rel = set(r.get("relevant_pages") or [])
        if not rel:
            continue
        true_topic = next(iter({page_topic[p] for p in rel if p in page_topic}))
        if not r["hits"]:
            continue
        pred_topic = page_topic.get(r["hits"][0]["page_id"], "?")
        if pred_topic in idx:
            mat[idx[true_topic], idx[pred_topic]] += 1
    if mat.sum() == 0:
        out.write_bytes(b"")
        return out
    # row-normalize
    mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(topics) + 2), max(4, 0.5 * len(topics) + 2)))
    im = ax.imshow(mat, cmap="Purples", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels(topics, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(topics, fontsize=8)
    for i in range(len(topics)):
        for j in range(len(topics)):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if mat[i, j] > 0.5 else "black",
            )
    ax.set_xlabel("retrieved page topic (top-1)")
    ax.set_ylabel("query topic")
    ax.set_title("Topic confusion matrix (row-normalized)")
    fig.colorbar(im, ax=ax, label="fraction")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
