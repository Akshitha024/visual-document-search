from __future__ import annotations

from cps.encoders.colpali_synth import encode_page, encode_query
from cps.scoring.maxsim import maxsim_one, maxsim_topk


def test_self_topic_scores_higher_than_other() -> None:
    page = encode_page("p1", "vaccines")
    q_self = encode_query("vaccines")
    q_other = encode_query("monetary policy")
    assert maxsim_one(q_self, page) > maxsim_one(q_other, page)


def test_topk_returns_k_hits() -> None:
    pages = [encode_page(f"p{i}", "climate") for i in range(5)]
    q = encode_query("climate")
    hits = maxsim_topk(q, pages, top_k=3)
    assert len(hits) == 3
    assert hits[0].rank == 1
    # scores should be descending
    from itertools import pairwise

    for a, b in pairwise(hits):
        assert a.score >= b.score
