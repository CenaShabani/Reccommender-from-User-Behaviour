"""Loading the RetailRocket dataset, plus a synthetic stand-in with the same schema.

RetailRocket (https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
ships three kinds of files:

``events.csv``
    ``timestamp`` (ms since epoch), ``visitorid``, ``event`` (``view`` /
    ``addtocart`` / ``transaction``), ``itemid``, ``transactionid`` (only set
    for purchases). About 2.76M rows from ~1.4M visitors over 4.5 months.
``item_properties_part1.csv`` / ``item_properties_part2.csv``
    A time-stamped key/value log of item properties. Almost every value is
    hashed; ``categoryid`` and ``available`` are the readable ones.
``category_tree.csv``
    ``categoryid`` -> ``parentid``.

The synthetic generator produces the same columns, so every function in this
package and every notebook runs unchanged on either source. It plants known
structure (user taste clusters, a popularity long tail, a view -> cart -> buy
funnel, new items launching over time, a handful of bots). That makes it a
useful sanity check: a personalised model *should* beat popularity on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

EVENT_COLUMNS = ["timestamp", "visitorid", "event", "itemid", "transactionid"]
EVENT_TYPES = ("view", "addtocart", "transaction")


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #
def raw_data_available(raw_dir: Path | str | None = None) -> bool:
    raw_dir = Path(raw_dir or config.RAW_DIR)
    return (raw_dir / "events.csv").exists()


def load_events(path: Path | str | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Load ``events.csv`` with compact dtypes and a proper datetime column."""
    path = Path(path or config.RAW_DIR / "events.csv")
    df = pd.read_csv(
        path,
        nrows=nrows,
        dtype={"visitorid": "int64", "itemid": "int64", "event": "category", "transactionid": "float64"},
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    validate_events(df)
    return df


def validate_events(df: pd.DataFrame) -> None:
    """Fail fast if a frame does not look like RetailRocket events."""
    missing = set(EVENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"events frame is missing columns: {sorted(missing)}")
    unknown = set(df["event"].astype(str).unique()) - set(EVENT_TYPES)
    if unknown:
        raise ValueError(f"unexpected event types: {sorted(unknown)}")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise TypeError("timestamp must be a datetime column (see load_events)")


def load_item_categories(raw_dir: Path | str | None = None, cache: bool = True) -> pd.DataFrame:
    """Return one ``categoryid`` per item (its latest value in the property log).

    The two property files hold ~20M rows, so we stream them in chunks and keep
    only the ``categoryid`` property. The result is cached as a small CSV.
    """
    raw_dir = Path(raw_dir or config.RAW_DIR)
    cache_path = config.PROCESSED_DIR / "item_categories.csv"
    if cache and cache_path.exists():
        return pd.read_csv(cache_path)

    parts = []
    for name in ("item_properties_part1.csv", "item_properties_part2.csv"):
        for chunk in pd.read_csv(raw_dir / name, chunksize=2_000_000):
            parts.append(chunk.loc[chunk["property"] == "categoryid", ["timestamp", "itemid", "value"]])
    props = pd.concat(parts, ignore_index=True)
    props = props.sort_values("timestamp").drop_duplicates("itemid", keep="last")
    out = pd.DataFrame({"itemid": props["itemid"].astype("int64"), "categoryid": props["value"].astype("int64")})
    out = out.sort_values("itemid").reset_index(drop=True)

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_path, index=False)
    return out


def load_category_tree(raw_dir: Path | str | None = None) -> pd.DataFrame:
    raw_dir = Path(raw_dir or config.RAW_DIR)
    return pd.read_csv(raw_dir / "category_tree.csv")


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def make_synthetic_events(
    n_users: int = 3000,
    n_items: int = 1500,
    n_clusters: int = 12,
    days: int = 60,
    taste_focus: float = 0.8,
    bot_fraction: float = 0.005,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a RetailRocket-shaped event log with known structure.

    Parameters
    ----------
    n_clusters
        Number of latent "taste" clusters. Each item belongs to one cluster,
        which doubles as its ``categoryid``.
    taste_focus
        Probability that a session is spent in one of the user's favourite
        clusters (rather than a random one). Higher means more predictable users.

    Returns
    -------
    events, item_categories
        ``events`` has the same columns and dtypes as :func:`load_events`.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2015-05-03")
    horizon = days * 86_400.0  # seconds

    # Items: a cluster, a base popularity drawn from a long-tailed distribution
    # and a launch time. A third of the catalogue launches during the window,
    # which gives "trending" signals something real to find.
    item_cluster = rng.integers(0, n_clusters, n_items)
    item_pop = rng.pareto(1.2, n_items) + 0.05
    item_launch = np.where(rng.random(n_items) < 0.67, -1.0, rng.uniform(0, horizon, n_items))
    cluster_items = [np.flatnonzero(item_cluster == c) for c in range(n_clusters)]

    # Users: one or two favourite clusters and a heavy-tailed activity level.
    n_bots = max(1, int(round(bot_fraction * n_users))) if bot_fraction > 0 else 0
    n_humans = n_users - n_bots
    n_favs = rng.integers(1, 3, n_humans)
    favs = [rng.choice(n_clusters, size=k, replace=False) for k in n_favs]
    n_sessions = np.clip(rng.lognormal(mean=1.0, sigma=0.9, size=n_humans).astype(int), 1, 60)

    ts_out, user_out, item_out, event_out = [], [], [], []

    def emit(ts, uid, items, evt):
        ts_out.append(ts)
        user_out.append(np.full(len(items), uid, dtype=np.int64))
        item_out.append(items)
        event_out.append(np.full(len(items), evt, dtype=object))

    for u in range(n_humans):
        session_times = np.sort(rng.uniform(0, horizon, n_sessions[u]))
        for t0 in session_times:
            cluster = rng.choice(favs[u]) if rng.random() < taste_focus else rng.integers(n_clusters)
            pool = cluster_items[cluster]
            pool = pool[item_launch[pool] <= t0]
            if len(pool) == 0:
                continue
            # Freshly launched items get a temporary boost (decays over ~a week).
            age_days = (t0 - np.maximum(item_launch[pool], 0)) / 86_400.0
            boost = 1.0 + 3.0 * np.exp(-age_days / 7.0) * (item_launch[pool] >= 0)
            w = item_pop[pool] * boost
            n_views = int(rng.integers(1, 7))
            viewed = rng.choice(pool, size=n_views, p=w / w.sum())
            view_ts = t0 + np.cumsum(rng.uniform(5, 120, n_views))
            emit(view_ts, u, viewed, "view")

            carted = rng.random(n_views) < 0.06
            if carted.any():
                cart_ts = view_ts[carted] + rng.uniform(5, 60, carted.sum())
                emit(cart_ts, u, viewed[carted], "addtocart")
                bought = rng.random(carted.sum()) < 0.4
                if bought.any():
                    emit(cart_ts[bought] + rng.uniform(30, 600, bought.sum()), u, viewed[carted][bought], "transaction")

    # Bots: thousands of uniformly random views, at all hours.
    for b in range(n_bots):
        n = int(rng.integers(1500, 3000))
        emit(np.sort(rng.uniform(0, horizon, n)), n_humans + b, rng.integers(0, n_items, n), "view")

    ts = np.concatenate(ts_out)
    events = pd.DataFrame(
        {
            "timestamp": start + pd.to_timedelta(ts, unit="s"),
            "visitorid": np.concatenate(user_out),
            "event": pd.Categorical(np.concatenate(event_out), categories=list(EVENT_TYPES)),
            "itemid": np.concatenate(item_out).astype(np.int64),
        }
    )
    # Hash-like ids so nobody accidentally relies on ids being 0..n-1.
    user_ids = rng.permutation(10 * n_users)[:n_users] + 1
    item_ids = rng.permutation(10 * n_items)[:n_items] + 1
    events["visitorid"] = user_ids[events["visitorid"].to_numpy()]
    events["itemid"] = item_ids[events["itemid"].to_numpy()]

    events = events.sort_values("timestamp", kind="stable").reset_index(drop=True)
    is_tx = events["event"] == "transaction"
    events["transactionid"] = np.nan
    events.loc[is_tx, "transactionid"] = np.arange(is_tx.sum(), dtype=float)
    events["timestamp"] = events["timestamp"].dt.floor("ms")
    events = events[EVENT_COLUMNS]

    item_categories = pd.DataFrame({"itemid": item_ids, "categoryid": item_cluster.astype(np.int64)})
    return events, item_categories.sort_values("itemid").reset_index(drop=True)


def load_or_synthesize(
    raw_dir: Path | str | None = None, verbose: bool = True, **synthetic_kwargs
) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    """Load the real RetailRocket data if present, otherwise fall back to synthetic.

    Returns ``(events, item_categories, source)`` where ``source`` is
    ``"retailrocket"`` or ``"synthetic"``. ``item_categories`` is ``None`` if
    the real property files are missing.
    """
    raw_dir = Path(raw_dir or config.RAW_DIR)
    if raw_data_available(raw_dir):
        events = load_events(raw_dir / "events.csv")
        has_props = (raw_dir / "item_properties_part1.csv").exists()
        cats = load_item_categories(raw_dir) if has_props else None
        if verbose:
            print(f"Loaded RetailRocket: {len(events):,} events from {raw_dir}")
        return events, cats, "retailrocket"

    events, cats = make_synthetic_events(**synthetic_kwargs)
    if verbose:
        print(
            "=" * 78 + "\n"
            "  USING SYNTHETIC DATA: data/raw/events.csv was not found.\n"
            "  The numbers below come from a simulator, NOT from RetailRocket.\n"
            "  Run `make data` (needs a Kaggle API token) to use the real dataset.\n" + "=" * 78
        )
    return events, cats, "synthetic"
