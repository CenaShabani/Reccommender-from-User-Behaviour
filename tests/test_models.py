import numpy as np
import pytest

from recsys.metrics import evaluate
from recsys.models import (
    ALSRecommender,
    BPRRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
    TrendingRecommender,
    top_k,
)


@pytest.fixture(scope="module")
def fitted(prepared):
    d = prepared
    kw = dict(train_events=d.train_events, item_map=d.item_map)
    return {
        "random": RandomRecommender().fit(d.X_train),
        "popularity": PopularityRecommender().fit(d.X_train),
        "trending": TrendingRecommender().fit(d.X_train, **kw),
        "item_knn": ItemKNNRecommender().fit(d.X_train),
        "als": ALSRecommender(iterations=8).fit(d.X_train),
        "bpr": BPRRecommender(epochs=20).fit(d.X_train),
    }


@pytest.fixture(scope="module")
def scores(fitted, prepared):
    return {n: evaluate(m, prepared.X_train, prepared.ground_truth)["ndcg@10"] for n, m in fitted.items()}


def test_top_k_is_sorted():
    s = np.array([[0.1, 0.9, 0.5, 0.7]])
    assert list(top_k(s, 3)[0]) == [1, 3, 2]


@pytest.mark.parametrize("name", ["random", "popularity", "trending", "item_knn", "als", "bpr"])
def test_recommendations_are_unique_and_unseen(fitted, prepared, name):
    X = prepared.X_train
    users = prepared.ground_truth.users[:50]
    recs = fitted[name].recommend(users, k=10)
    assert recs.shape == (len(users), 10)
    for u, row in zip(users, recs):
        assert len(set(row)) == 10
        assert (np.asarray(X[u, row].todense()).ravel() == 0).all()


def test_popularity_beats_random(scores):
    assert scores["popularity"] > 2 * scores["random"]


@pytest.mark.parametrize("name", ["item_knn", "als", "bpr"])
def test_personalised_models_beat_popularity(scores, name):
    # The synthetic world has planted taste clusters, so personalisation must help.
    assert scores[name] > scores["popularity"] * 1.3


def test_item_knn_similar_items_share_a_category(fitted, prepared, synthetic):
    _, cats = synthetic
    cat_of = dict(zip(cats["itemid"], cats["categoryid"]))
    model, item_map = fitted["item_knn"], prepared.item_map
    agree = []
    for item in range(0, prepared.X_train.shape[1], 25):
        nbrs, _ = model.similar_items(item, n=5)
        if len(nbrs):
            agree.append(np.mean([cat_of[r] == cat_of[item_map.to_raw(item)] for r in item_map.to_raw(nbrs)]))
    assert np.mean(agree) > 0.5


def test_item_knn_explanations_come_from_history(fitted, prepared):
    model = fitted["item_knn"]
    u = int(prepared.ground_truth.users[0])
    item = int(model.recommend([u], k=1)[0, 0])
    history = set(prepared.X_train[u].indices)
    assert {j for j, _ in model.explain(u, item)} <= history


def test_als_loss_decreases(prepared):
    m = ALSRecommender(iterations=5, verbose=True).fit(prepared.X_train)
    assert all(a >= b - 1e-6 for a, b in zip(m.loss_history_, m.loss_history_[1:]))


def test_bpr_learns_to_order(fitted):
    auc = fitted["bpr"].auc_history_
    assert auc[-1] > 0.85 and auc[-1] > auc[0]
