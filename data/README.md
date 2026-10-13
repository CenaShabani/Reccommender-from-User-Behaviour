# Data

This project uses the **RetailRocket e-commerce dataset** from Kaggle:
<https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset> (licence: CC BY-NC-SA 4.0).

Nothing in `raw/` or `processed/` is committed to git.

## Getting it

```bash
pip install kaggle
# 1. Create an API token: kaggle.com → Settings → "Create New Token" → save as ~/.kaggle/kaggle.json (chmod 600)
# 2. Open the dataset page once in a browser and accept its terms
make data            # = bash scripts/download_data.sh
```

Or download the zip manually from the Kaggle page and unzip it into `data/raw/`.

Expected layout:

```
data/raw/
├── events.csv                   # ~2.76M rows, ~90 MB
├── item_properties_part1.csv    # ~11M rows
├── item_properties_part2.csv    # ~9M rows
└── category_tree.csv
```

Only `events.csv` is required. The property files enable category features (notebooks 01, 04–07); without them those parts are skipped.

## Schema

### `events.csv`

| column | type | notes |
|---|---|---|
| `timestamp` | int (ms since epoch) | converted to `datetime64` by `recsys.data.load_events` |
| `visitorid` | int | anonymised visitor id |
| `event` | str | `view`, `addtocart` or `transaction` |
| `itemid` | int | anonymised item id |
| `transactionid` | float | set only for `transaction` rows |

### `item_properties_part*.csv`

A time-stamped key/value log: `timestamp, itemid, property, value`. Almost all properties and values are hashed. The readable ones are `categoryid` and `available`. `recsys.data.load_item_categories` keeps the latest `categoryid` per item and caches the result to `processed/item_categories.csv`.

### `category_tree.csv`

`categoryid, parentid`: the category hierarchy.

## No Kaggle access? Synthetic data

If `raw/events.csv` is missing, every notebook and script falls back to `recsys.data.make_synthetic_events`, which produces the **same columns** with planted structure:

* taste clusters, which double as categories;
* a long-tailed popularity distribution;
* a view → cart → purchase funnel;
* items launching during the window;
* a few bots.

It is perfect for learning the mechanics and for tests. A loud banner reminds you that the numbers are not real.
