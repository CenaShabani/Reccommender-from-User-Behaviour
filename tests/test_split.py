import numpy as np

from recsys.split import leave_last_out, random_split, temporal_split


def test_temporal_split_has_no_time_leakage(synthetic):
    events = synthetic[0]
    s = temporal_split(events, test_days=7)
    assert s.train["timestamp"].max() < s.cutoff <= s.test["timestamp"].min()
    assert len(s.train) + len(s.test) == len(events)


def test_ground_truth_excludes_seen_and_cold(prepared):
    X, gt = prepared.X_train, prepared.ground_truth
    assert len(gt) > 0
    for u, items in gt.relevant.items():
        assert 0 <= u < X.shape[0]
        assert (items >= 0).all() and (items < X.shape[1]).all()
        assert (np.asarray(X[u, items].todense()).ravel() == 0).all()
    assert gt.stats["cold_users"] >= 0


def test_train_matrix_only_uses_train_period(prepared):
    assert prepared.train_events["timestamp"].max() < prepared.cutoff


def test_random_split_sizes(synthetic):
    tr, te = random_split(synthetic[0], test_frac=0.2)
    assert abs(len(te) / (len(tr) + len(te)) - 0.2) < 0.02


def test_leave_last_out_takes_latest_event(synthetic):
    tr, te = leave_last_out(synthetic[0])
    assert te["visitorid"].is_unique
    last_train = tr.groupby("visitorid")["timestamp"].max()
    common = te.set_index("visitorid").loc[last_train.index, "timestamp"]
    assert (common.to_numpy() >= last_train.to_numpy()).all()
