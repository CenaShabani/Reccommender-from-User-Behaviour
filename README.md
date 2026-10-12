# Recommender Systems from User Behaviour

**A hands-on learning project:** build, evaluate and reason about a product recommender trained on real e-commerce clickstream data. It goes from a popularity baseline all the way to a two-stage retrieval and ranking system, the architecture most production recommenders use.

The project is written the way a Principal Data Scientist would mentor a team. Every step explains **why**, every pitfall is called out, and every notebook ends with exercises.

* **Dataset:** [RetailRocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) (Kaggle). About 2.76M real `view` / `addtocart` / `transaction` events from ~1.4M visitors over 4.5 months.
* **Task:** for each returning visitor, recommend 10 products they have not interacted with yet.
* **Stack:** numpy, pandas, scipy, scikit-learn and matplotlib only. Every algorithm is implemented from scratch in readable code, with no black-box recommender library.

---

## Learning path

Work through the notebooks in order. Each one has learning objectives, theory (with the maths), code, a "Principal DS note" on real-world pitfalls, key takeaways and graded exercises.

| # | Notebook | You will learn |
|---|---|---|
| 00 | [Problem framing](notebooks/00_problem_framing.ipynb) | Explicit vs implicit feedback; offline → online metric hierarchy; scoping questions |
| 01 | [Exploratory data analysis](notebooks/01_eda.ipynb) | Funnel, sparsity, long tail, bots, repeat behaviour, and turning each finding into a decision |
| 02 | [Preprocessing & splits](notebooks/02_preprocessing_and_splits.ipynb) | Event weighting, k-core, sparse matrices, **temporal splits**, a leakage experiment, cold start |
| 03 | [Baselines & metrics](notebooks/03_baselines_and_metrics.ipynb) | Precision/recall/MAP/NDCG by hand; random, popularity and trending baselines; coverage and novelty |
| 04 | [Item-kNN](notebooks/04_item_knn.ipynb) | Cosine similarity, shrinkage, BM25, **validation-based tuning**, explanations |
| 05 | [Matrix factorisation](notebooks/05_matrix_factorization.ipynb) | Implicit ALS (derivation + implementation), BPR, hyperparameter search, embeddings |
| 06 | [Two-stage ranking](notebooks/06_two_stage_ranking.ipynb) | Candidate generation, candidate recall, leakage-free ranker labels, GBDT ranker, feature importance |
| 07 | [Evaluation & production](notebooks/07_evaluation_and_production.ipynb) | Paired bootstrap, segments, popularity bias, cold-start policy, A/B test sizing, architecture, monitoring |

Roughly 6–10 hours end to end, plus exercises.

**Prerequisites:** comfortable with Python and pandas, basic linear algebra (dot products, matrix multiplication) and basic probability. No prior recommender-systems knowledge is needed.

---

## Quick start

```bash
git clone <this repo> && cd Reccommender-from-User-Behaviour
python -m venv .venv && source .venv/bin/activate
make setup          # pip install -e ".[dev,notebooks,data]"

make test           # unit tests on synthetic data (~10 s)
make data           # download RetailRocket from Kaggle (needs an API token; see data/README.md)
make benchmark      # fit and evaluate every model, then write results/leaderboard.{csv,md}

jupyter lab notebooks/
```

**No Kaggle account?** Everything still runs. Without `data/raw/events.csv`, the code falls back to a **synthetic** event log with the same schema and planted structure, and prints a loud banner so you never mistake it for real results. Use it to learn the mechanics, then switch to the real data for real insight.

---

## Project layout

```
├── notebooks/              # the learning path (committed without outputs)
├── src/recsys/             # the reusable, tested package the notebooks import
│   ├── config.py           # paths, event weights, seeds, defaults
│   ├── data.py             # load RetailRocket; synthetic generator with the same schema
│   ├── preprocess.py       # cleaning, weights, bot filter, k-core, IdMapper, sparse matrix
│   ├── split.py            # temporal / random / leave-last-out splits; ground truth
│   ├── pipeline.py         # prepare_data(): events → train matrix + ground truth, in the right order
│   ├── metrics.py          # precision, recall, hit-rate, MAP, NDCG, MRR, coverage, novelty
│   ├── plotting.py         # consistent, colour-blind-safe chart style
│   └── models/
│       ├── base.py         # Recommender interface: fit / score / recommend
│       ├── popularity.py   # random, popularity, trending (time-decayed)
│       ├── itemknn.py      # item-item cosine with BM25, shrinkage and top-K pruning
│       ├── als.py          # implicit ALS (Hu, Koren & Volinsky 2008), from scratch
│       ├── bpr.py          # BPR-MF (Rendle et al. 2009), vectorised SGD
│       └── two_stage.py    # candidate generation + gradient-boosted ranker
├── tests/                  # pytest suite: metrics, leakage invariants, model sanity checks
├── scripts/
│   ├── download_data.sh    # Kaggle download
│   ├── run_benchmark.py    # end-to-end leaderboard
│   └── execute_notebooks.py
└── data/                   # raw/ and processed/ are git-ignored; see data/README.md
```

**Read the source as well as the notebooks.** Each module's docstring explains the reasoning behind it, and the models are short enough to read in one sitting (ALS is ~40 lines of real code).

---

## The models

| Model | Family | Idea | Strength | Weakness |
|---|---|---|---|---|
| Random | baseline | uniform scores | sanity floor | useless |
| Popularity | baseline | most distinct users | very strong on long-tailed data | same list for everyone, tiny coverage |
| Trending | baseline | popularity with exponential time decay | adapts to new and seasonal items | still not personal |
| Item-kNN | neighbourhood CF | items co-interacted by the same users are similar | fast, explainable, great for short histories | can't generalise beyond co-occurrence |
| ALS | matrix factorisation | weighted least squares on preference + confidence | generalises; embeddings reusable | needs tuning; less explainable |
| BPR | matrix factorisation | pairwise ranking loss with SGD | optimises order directly | sensitive to sampling and learning rate |
| Two-stage | hybrid | union of generators → GBDT ranker on rich features | combines strengths; easy to add business signals | more moving parts; labels must avoid leakage |

## Evaluation protocol

1. Remove bots (users with more than 1,000 events).
2. **Temporal split:** train on everything before the cutoff, test on the last 7 days.
3. k-core filter (≥ 2) on the **training** period only.
4. Ground truth: each returning user's test-week items that they had **not** interacted with in training. Cold-start users and items are counted and reported, not hidden.
5. Recommend 10 unseen items per user, and report NDCG@10 (primary), recall, hit-rate, MAP, MRR, coverage and novelty.
6. Tune hyperparameters on a **validation** window carved out of training. Touch the test window once.

## Results

Run `make benchmark` to produce `results/leaderboard.md` for your data. For reference, this is the **synthetic** dataset (not RetailRocket, so don't quote these numbers), with default hyperparameters:

| model | NDCG@10 | recall@10 | hit-rate@10 | coverage |
|---|---|---|---|---|
| two_stage | 0.201 | 0.267 | 0.507 | 0.33 |
| item_knn | 0.193 | 0.257 | 0.463 | 0.40 |
| als | 0.172 | 0.232 | 0.432 | 0.38 |
| bpr | 0.160 | 0.218 | 0.439 | 0.36 |
| trending | 0.074 | 0.126 | 0.286 | 0.02 |
| popularity | 0.066 | 0.104 | 0.257 | 0.02 |
| random | 0.008 | 0.013 | 0.034 | 1.00 |

Once you have run it on RetailRocket, paste your own table here. That is exercise zero.

---

## Principal DS principles this project teaches

1. **Frame before you model.** Define the decision, the user and the success metric first.
2. **Beat the right baseline.** Beat popularity or trending, not random.
3. **Split by time.** Anything else leaks the future.
4. **Never tune on test.** Use a validation window.
5. **Look past the average.** Check segments, confidence intervals, coverage and popularity bias.
6. **Policies, not just models.** Cold-start fallbacks and business filters are part of the product.
7. **Offline picks candidates, online picks winners.** A/B test before you ship.

## Further reading

* Hu, Koren & Volinsky (2008). *Collaborative Filtering for Implicit Feedback Datasets.* ICDM.
* Rendle et al. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback.* UAI.
* Linden, Smith & York (2003). *Amazon.com Recommendations: Item-to-Item Collaborative Filtering.* IEEE Internet Computing.
* Covington, Adams & Sargin (2016). *Deep Neural Networks for YouTube Recommendations.* RecSys. (The two-stage architecture.)
* Steck (2019). *Embarrassingly Shallow Autoencoders for Sparse Data* (EASE). WWW.
* Dacrema, Cremonesi & Jannach (2019). *Are We Really Making Much Progress?* RecSys. (Why strong baselines matter.)

## Licence

Code: MIT (see `LICENSE`). The RetailRocket data is distributed by Kaggle under CC BY-NC-SA 4.0 and is not included in this repository.
