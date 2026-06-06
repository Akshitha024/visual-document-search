"""Synthetic ColPali-shaped encoder.

The real ColPali setup runs a PaliGemma-style VLM over page images and
produces (n_patches, embed_dim) per page. For CI and laptop demos we
substitute a deterministic generator that:

  - takes a page's "topic" (a small string label)
  - produces patches that include some topic-correlated coordinates +
    random noise

The MaxSim retriever then has something it can plausibly rank. Real-data
mode swaps this for `vidore/colpali` weights.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from ..types import PageEmbedding

DEFAULT_EMBED_DIM = 64
DEFAULT_N_PATCHES = 32


def _topic_to_anchor(topic: str, embed_dim: int) -> NDArray[np.float64]:
    """Deterministic anchor vector per topic string."""
    h = hashlib.sha256(topic.encode()).digest()
    seed = int.from_bytes(h[:4], byteorder="big") & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(embed_dim).astype(np.float64)
    v /= float(max(float(np.linalg.norm(v)), 1e-12))
    return v


def encode_page(
    page_id: str,
    topic: str,
    embed_dim: int = DEFAULT_EMBED_DIM,
    n_patches: int = DEFAULT_N_PATCHES,
    topic_strength: float = 0.6,
    seed: int = 0,
) -> PageEmbedding:
    anchor = _topic_to_anchor(topic, embed_dim)
    rng = np.random.default_rng(hash((page_id, seed)) & 0xFFFFFFFF)
    noise = rng.standard_normal((n_patches, embed_dim)) * 0.3
    patches = topic_strength * anchor[None, :] + (1 - topic_strength) * noise
    # L2-normalize each patch (ColPali normalizes per-patch)
    norms_p = np.maximum(np.linalg.norm(patches, axis=1, keepdims=True), 1e-12)
    patches = patches / norms_p
    return PageEmbedding(page_id=page_id, patches=patches.astype(np.float64))


def encode_query(
    text: str,
    embed_dim: int = DEFAULT_EMBED_DIM,
    n_tokens: int = 8,
    topic_strength: float = 0.7,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Query side encoder. Each query token is also a (d,)-vector; we
    return (n_tokens, d) so MaxSim is meaningful.
    """
    anchor = _topic_to_anchor(text, embed_dim)
    rng = np.random.default_rng(hash((text, "q", seed)) & 0xFFFFFFFF)
    noise = rng.standard_normal((n_tokens, embed_dim)) * 0.3
    toks = topic_strength * anchor[None, :] + (1 - topic_strength) * noise
    norms_t = np.maximum(np.linalg.norm(toks, axis=1, keepdims=True), 1e-12)
    toks = toks / norms_t
    result: NDArray[np.float64] = toks.astype(np.float64)
    return result
