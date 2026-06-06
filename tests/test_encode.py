from __future__ import annotations

import numpy as np

from cps.encoders.colpali_synth import encode_page, encode_query


def test_encode_page_shape() -> None:
    e = encode_page("p1", "vaccines")
    assert e.page_id == "p1"
    assert e.patches.shape[0] > 0
    assert e.patches.shape[1] > 0


def test_encode_page_normalized_patches() -> None:
    e = encode_page("p1", "climate")
    norms = np.linalg.norm(e.patches, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-6)


def test_encode_query_shape() -> None:
    q = encode_query("anything")
    assert q.shape[0] > 0


def test_topic_affects_anchor() -> None:
    a = encode_query("vaccines")
    b = encode_query("monetary policy")
    # different topics produce different anchors so mean cosine across tokens
    # should be much less than self-similarity
    self_sim = float((a @ a.T).mean())
    cross_sim = float((a @ b.T).mean())
    assert self_sim > cross_sim
