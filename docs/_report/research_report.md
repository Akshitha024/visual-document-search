---
title: "visual-document-search: ColPali-style multimodal page retrieval without OCR"
author: "Akshitha Reddy Lingampally"
date: "2026-06-06"
geometry: margin=1in
fontsize: 11pt
---

# Abstract

We present `visual-document-search`, a minimal ColPali-style retrieval
harness. Each page is encoded into per-patch visual embeddings and each
query into per-token text embeddings; the score for a (query, page)
pair is the MaxSim aggregation that ColBERT pioneered for text. The
package ships a deterministic synthetic encoder so the suite runs
without a Vision-Language Model, with the interface matching what the
real `vidore/colpali` backbone produces. We report a synthetic-corpus
benchmark (25 queries × 100 pages across 5 topics): page_hit@5 = 0.24,
doc_hit@10 = 0.36, MRR@10 = 0.13 — exactly the shape we'd expect with a
weak (synthetic) encoder, while showing the chart-suite (topic confusion,
score margin, per-topic accuracy) works end-to-end.

# 1. Background

Traditional document RAG runs OCR first, then text retrieval. This
loses two things: layout information (where on the page the text was)
and visual content (figures, charts, scanned tables). ColPali (Faysse
et al., 2024) showed that running a Vision-Language Model directly on
the page image produces per-patch embeddings that, when scored with
ColBERT's MaxSim aggregation, beat OCR + text retrieval on document
benchmarks like ViDoRe.

The architecture is conceptually simple: VLM encoder per page → list
of per-patch embeddings; text encoder per query → list of per-token
embeddings; score = sum over query tokens of max over patches of
dot product.

# 2. Related Work

- **ColPali** (Faysse et al., 2024): the paper this project follows.
- **ColBERT** (Khattab & Zaharia, 2020): the MaxSim aggregation
  pattern. ColPali is the visual extension.
- **PaliGemma** (Beyer et al., 2024): the underlying VLM ColPali
  uses; we use a generic interface so it's swappable.

# 3. Method

## 3.1 Synthetic encoder

For testing without a VLM, we generate per-page patches as
deterministic topic-anchored vectors plus noise:

```
anchor = hash_to_unit_vector(topic)
patches = topic_strength * anchor + (1 - topic_strength) * noise
patches /= ||patches||   # per-patch L2 normalize
```

`topic_strength = 0.6` controls signal-to-noise. The query encoder is
the same shape: anchor for the query's topic + noise per query token.

## 3.2 MaxSim scoring

```
score(query, page) = sum over q in query_tokens of
                     max over p in page_patches of <q, p>
```

L2-normalized inputs make the dot products cosine similarities.

## 3.3 Metrics

- `page_hit@k`: 1 if any of the top-k pages is a relevant page (the
  strictest metric)
- `doc_hit@k`: 1 if any of the top-k pages is from a relevant
  document (more forgiving when a doc has multiple pages)
- `recall@k`: fraction of relevant pages in top-k
- `mrr@k`: 1 / rank of first relevant page

# 4. Data

In-repo synthetic corpus: 5 topics × 5 documents per topic × 4 pages
per doc = 100 pages. 25 queries, one per (topic, doc). Each query
targets a specific (topic, doc) pair; the 4 pages of that doc are the
relevant set.

Real-data mode: swap the synthetic encoder for `vidore/colpali` and
plug into a ViDoRe-style PDF benchmark.

# 5. Evaluation Setup

Standard run: 25 queries × top-10 retrieval over 100 pages.
Hardware: Apple M-series CPU.

# 6. Results

| metric          | value |
|-----------------|------:|
| page_hit@1      | 0.080 |
| page_hit@5      | 0.240 |
| page_hit@10     | 0.360 |
| doc_hit@5       | 0.240 |
| recall@10       | 0.150 |
| mrr@10          | 0.131 |

The synthetic encoder is intentionally weak (topic_strength = 0.6
plus heavy noise), so the chart *shapes* are interesting, not the
absolute numbers. Real ColPali on a real PDF benchmark lands
page_hit@5 around 0.70-0.85 on ViDoRe (per Faysse et al., 2024);
that gap is the value of the trained backbone over the topic-anchored
synthetic stand-in.

# 7. Ablations

`topic_strength` sweep ∈ {0.4, 0.6, 0.8}: page_hit@5 lifts from
0.08 (very noisy) to 0.60 (clean topic signal); a sanity check that
the retrieval pipeline reflects encoder quality.

# 8. Discussion

The harness's value is its end-to-end shape, not the synthetic
numbers. Once a real VLM is plugged in, the same five charts surface
the things that matter for production document retrieval: per-topic
accuracy (the "which subject matter the model is good at"), score
margin (the "how confident the rank-1 was"), topic confusion (the
"which subject matter gets confused with which"). All five run on the
same per-query JSONL artifact, so swapping encoders re-uses the chart
code unchanged.

# 9. Limitations

1. **Synthetic encoder.** Real ColPali needs `colpali-engine` +
   a GPU. The harness is set up to drop it in but we don't ship
   the integration.
2. **No PDF → page-image pipeline.** Production needs
   `pdf2image` + PIL.
3. **Brute-force MaxSim.** Scales O(corpus × patches × q_tokens);
   above ~10K pages use ColBERTv2-style PLAID indexing.
4. **Page-level retrieval only.** Doc-level aggregation is "any page
   = doc hit"; a weighted-sum-per-doc variant would lift doc_hit
   on long-doc corpora.

# 10. Future Work

- [ ] Real ColPali backbone (`vidore/colpali`) + ViDoRe benchmark.
- [ ] PLAID indexing (centroids + IVF) for scale.
- [ ] Multi-page query (question spans multiple pages).
- [ ] Compare against OCR + BM25 baseline (the ColPali headline gap).

# 11. References

- Beyer, L., et al. (2024). *PaliGemma: A versatile 3B VLM for
  transfer.* arXiv:2407.07726.
- Faysse, M., et al. (2024). *ColPali: Efficient Document Retrieval
  with Vision Language Models.* arXiv:2407.01449.
- Khattab, O., & Zaharia, M. (2020). *ColBERT: Efficient and
  Effective Passage Search via Contextualized Late Interaction
  over BERT.* SIGIR.

# Appendix A. Reproducibility

- Repo: `Akshitha024/visual-document-search`, MIT.
- Reproduce: `make bench && make plots`.
- 5 charts in `results/figures/`.
- Test artifacts in `docs/test_results/`.
