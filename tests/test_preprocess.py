import numpy as np
import pandas as pd

from recsys.preprocess import (
    IdMapper,
    add_weights,
    build_interaction_matrix,
    clean_events,
    filter_outlier_users,
    k_core_filter,
)


def _toy():
    t = pd.Timestamp("2020-01-01")
    rows = [
        (t, 1, "view", 10), (t, 1, "view", 10), (t, 1, "addtocart", 10), (t, 1, "view", 11),
        (t, 2, "view", 10), (t, 2, "view", 11), (t, 2, "transaction", 12),
        (t, 3, "view", 13),  # user 3 and item 13 are too sparse for a 2-core
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "visitorid", "event", "itemid"])
    df["transactionid"] = np.nan
    return df


def test_clean_drops_exact_duplicates():
    assert len(clean_events(_toy())) == len(_toy()) - 1


def test_weights_follow_config():
    w = add_weights(_toy(), {"view": 1, "addtocart": 4, "transaction": 8})
    assert w.loc[w["event"] == "transaction", "weight"].iloc[0] == 8


def test_k_core_invariant(synthetic):
    events = synthetic[0]
    core = k_core_filter(events, min_user_items=3, min_item_users=3)
    pairs = core[["visitorid", "itemid"]].drop_duplicates()
    assert pairs["visitorid"].value_counts().min() >= 3
    assert pairs["itemid"].value_counts().min() >= 3


def test_k_core_on_toy():
    core = k_core_filter(_toy(), 2, 2)
    assert set(core["visitorid"]) == {1, 2}
    assert set(core["itemid"]) == {10, 11}


def test_outlier_filter_removes_bots(synthetic):
    events = synthetic[0]
    filtered, removed = filter_outlier_users(events, max_events=1000)
    assert len(removed) > 0
    assert filtered["visitorid"].value_counts().max() <= 1000


def test_id_mapper_roundtrip():
    m = IdMapper([30, 10, 30, 20])
    assert len(m) == 3
    idx = m.to_index([10, 20, 30, 99])
    assert idx[-1] == -1
    assert list(m.to_raw(idx[:3])) == [10, 20, 30]


def test_interaction_matrix_sums_then_transforms():
    ev = add_weights(clean_events(_toy()), {"view": 1, "addtocart": 4, "transaction": 8})
    um, im = IdMapper(ev["visitorid"]), IdMapper(ev["itemid"])
    X = build_interaction_matrix(ev, um, im, transform="none")
    u1, i10 = um.to_index([1])[0], im.to_index([10])[0]
    assert X[u1, i10] == 1 + 4  # duplicate view dropped by clean_events, then view + cart
    Xl = build_interaction_matrix(ev, um, im, transform="log1p")
    assert np.isclose(Xl[u1, i10], np.log1p(5))
    Xb = build_interaction_matrix(ev, um, im, transform="binary")
    assert set(Xb.data) == {1.0}
