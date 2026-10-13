import pandas as pd
import pytest

from recsys.data import EVENT_COLUMNS, EVENT_TYPES, load_events, make_synthetic_events, validate_events


def test_synthetic_matches_retailrocket_schema(synthetic):
    events, cats = synthetic
    assert list(events.columns) == EVENT_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(events["timestamp"])
    assert set(events["event"].astype(str)) <= set(EVENT_TYPES)
    validate_events(events)
    # transactionid is set exactly on purchases
    is_tx = events["event"] == "transaction"
    assert events.loc[is_tx, "transactionid"].notna().all()
    assert events.loc[~is_tx, "transactionid"].isna().all()
    assert set(cats.columns) == {"itemid", "categoryid"}
    assert set(events["itemid"]) <= set(cats["itemid"])


def test_synthetic_has_a_funnel(synthetic):
    counts = synthetic[0]["event"].value_counts()
    assert counts["view"] > counts["addtocart"] > counts["transaction"] > 0


def test_synthetic_is_deterministic():
    a, _ = make_synthetic_events(n_users=200, n_items=100, seed=3)
    b, _ = make_synthetic_events(n_users=200, n_items=100, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_load_events_roundtrip(tmp_path, synthetic):
    events = synthetic[0].head(500).copy()
    raw = events.assign(timestamp=events["timestamp"].astype("int64") // 1_000_000)
    raw.to_csv(tmp_path / "events.csv", index=False)
    loaded = load_events(tmp_path / "events.csv")
    pd.testing.assert_series_equal(loaded["timestamp"], events["timestamp"], check_dtype=False)
    assert (loaded["visitorid"].to_numpy() == events["visitorid"].to_numpy()).all()


def test_validate_rejects_unknown_events(synthetic):
    bad = synthetic[0].head(10).copy()
    bad["event"] = bad["event"].astype(str)
    bad.loc[bad.index[0], "event"] = "click"
    with pytest.raises(ValueError):
        validate_events(bad)
