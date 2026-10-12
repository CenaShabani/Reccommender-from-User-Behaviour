import numpy as np
import pytest

from recsys.metrics import (
    average_precision_at_k,
    catalog_coverage,
    hit_rate_at_k,
    hits_vector,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Recommended [a, b, c, d, e], relevant {b, e, z}: hits at ranks 2 and 5.
RECS = np.array([1, 2, 3, 4, 5])
REL = np.array([2, 5, 99])


def test_hand_computed_values():
    h = hits_vector(RECS, REL)
    assert list(h) == [0, 1, 0, 0, 1]
    assert precision_at_k(h, 5) == pytest.approx(2 / 5)
    assert recall_at_k(h, 3, 5) == pytest.approx(2 / 3)
    assert hit_rate_at_k(h, 1) == 0.0 and hit_rate_at_k(h, 2) == 1.0
    # AP: precision at each hit = 1/2 and 2/5, averaged over min(|rel|, k) = 3
    assert average_precision_at_k(h, 3, 5) == pytest.approx((1 / 2 + 2 / 5) / 3)
    dcg = 1 / np.log2(3) + 1 / np.log2(6)
    idcg = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert ndcg_at_k(h, 3, 5) == pytest.approx(dcg / idcg)
    assert reciprocal_rank(h) == pytest.approx(1 / 2)


def test_perfect_list_scores_one():
    h = hits_vector(np.array([7, 8]), np.array([7, 8]))
    assert ndcg_at_k(h, 2, 2) == pytest.approx(1.0)
    assert average_precision_at_k(h, 2, 2) == pytest.approx(1.0)


def test_no_hits_scores_zero():
    h = hits_vector(np.array([1, 2, 3]), np.array([9]))
    assert ndcg_at_k(h, 1, 3) == 0 and average_precision_at_k(h, 1, 3) == 0 and reciprocal_rank(h) == 0


def test_coverage_and_novelty():
    recs = np.array([[0, 1], [0, 2]])
    assert catalog_coverage(recs, 10) == pytest.approx(0.3)
    pop = np.array([100, 1, 1, 0])
    assert novelty(np.array([[1, 2]]), pop, 100) > novelty(np.array([[0, 0]]), pop, 100)
