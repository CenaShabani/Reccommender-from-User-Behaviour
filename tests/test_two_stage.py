import numpy as np
import pytest

from recsys.metrics import evaluate
from recsys.models import ALSRecommender, ItemKNNRecommender, PopularityRecommender, TwoStageRecommender
from recsys.models.two_stage import generate_candidates


@pytest.fixture(scope="module")
def two_stage(prepared, synthetic):
    d = prepared
    return TwoStageRecommender(n_candidates=30, val_days=7).fit(
        d.X_train, train_events=d.train_events, user_map=d.user_map, item_map=d.item_map, item_categories=synthetic[1]
    )


def test_fit_requires_events(prepared):
    with pytest.raises(ValueError):
        TwoStageRecommender().fit(prepared.X_train)


def test_ranker_training_report(two_stage):
    r = two_stage.train_report_
    assert r["ranker_rows"] > 0 and 0 < r["positive_rate"] < 1
    assert 0 < r["candidate_recall"] <= 1
    assert "user_category_affinity" in two_stage.feature_names_


def test_candidates_are_unique_and_unseen(prepared):
    X = prepared.X_train
    gens = {"pop": PopularityRecommender().fit(X), "knn": ItemKNNRecommender().fit(X)}
    users = prepared.ground_truth.users[:40]
    c = generate_candidates(users, X, gens, n_candidates=20)
    assert not c.duplicated(["user", "item"]).any()
    assert (np.asarray(X[c["user"].to_numpy(), c["item"].to_numpy()]).ravel() == 0).all()
    assert c.groupby("user").size().min() >= 20


def test_two_stage_is_competitive_with_its_generators(two_stage, prepared):
    d = prepared
    ndcg = lambda m: evaluate(m, d.X_train, d.ground_truth)["ndcg@10"]
    best_generator = max(ndcg(ItemKNNRecommender().fit(d.X_train)), ndcg(ALSRecommender().fit(d.X_train)))
    assert ndcg(two_stage) >= 0.95 * best_generator
