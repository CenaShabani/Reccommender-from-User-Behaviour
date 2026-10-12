"""End-to-end benchmark: load -> prepare -> fit every model -> leaderboard.

    python scripts/run_benchmark.py              # real data if present, else synthetic
    python scripts/run_benchmark.py --synthetic  # force synthetic
    python scripts/run_benchmark.py --models popularity item_knn als

Writes results/leaderboard.csv and results/leaderboard.md.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from recsys import config
from recsys.data import load_or_synthesize, make_synthetic_events
from recsys.metrics import evaluate
from recsys.models import (
    ALSRecommender,
    BPRRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
    TrendingRecommender,
    TwoStageRecommender,
)
from recsys.pipeline import prepare_data

MODELS = {
    "random": RandomRecommender,
    "popularity": PopularityRecommender,
    "trending": TrendingRecommender,
    "item_knn": ItemKNNRecommender,
    "als": ALSRecommender,
    "bpr": BPRRecommender,
    "two_stage": TwoStageRecommender,
}


def to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids an extra dependency on ``tabulate``)."""
    header = "| model | " + " | ".join(df.columns) + " |"
    sep = "|---" * (len(df.columns) + 1) + "|"
    body = ["| " + str(i) + " | " + " | ".join(str(v) for v in row) + " |" for i, *row in df.itertuples()]
    return "\n".join([header, sep, *body])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true", help="use synthetic data even if the real data exists")
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--test-days", type=float, default=config.TEST_DAYS)
    p.add_argument("--k", type=int, default=config.K)
    p.add_argument("--max-users", type=int, default=config.MAX_EVAL_USERS)
    p.add_argument("--target", choices=["any", "cart_or_buy"], default="any", help="which test events count as relevant")
    args = p.parse_args()

    if args.synthetic:
        events, cats = make_synthetic_events()
        source = "synthetic"
    else:
        events, cats, source = load_or_synthesize()

    target = None if args.target == "any" else ("addtocart", "transaction")
    t0 = time.time()
    data = prepare_data(events, test_days=args.test_days, target_events=target)
    print(f"\nData prepared in {time.time() - t0:.1f}s\n{data.summary().to_string()}\n")

    fit_kwargs = dict(train_events=data.train_events, user_map=data.user_map, item_map=data.item_map, item_categories=cats)
    rows = {}
    for name in args.models:
        t0 = time.time()
        model = MODELS[name]().fit(data.X_train, **fit_kwargs)
        fit_s = time.time() - t0
        t0 = time.time()
        rows[name] = evaluate(model, data.X_train, data.ground_truth, k=args.k, max_users=args.max_users)
        rows[name].update(fit_seconds=round(fit_s, 1), eval_seconds=round(time.time() - t0, 1))
        print(f"{name:>12}: ndcg@{args.k}={rows[name][f'ndcg@{args.k}']:.4f}  (fit {fit_s:.1f}s)")

    board = pd.DataFrame(rows).T.sort_values(f"ndcg@{args.k}", ascending=False)
    board["n_users"] = board["n_users"].astype(int)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    board.to_csv(config.RESULTS_DIR / "leaderboard.csv")
    md = f"Data source: **{source}**, target: `{args.target}`, cutoff {data.cutoff:%Y-%m-%d}\n\n" + to_markdown(board.round(4))
    (config.RESULTS_DIR / "leaderboard.md").write_text(md + "\n")
    print("\n" + board.round(4).to_string())
    print(f"\nSaved to {config.RESULTS_DIR}/leaderboard.{{csv,md}}")


if __name__ == "__main__":
    main()
