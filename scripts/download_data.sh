#!/usr/bin/env bash
# Download the RetailRocket e-commerce dataset from Kaggle into data/raw/.
#
# Prerequisites (one-off):
#   1. pip install kaggle            (or: pip install -e ".[data]")
#   2. Create an API token at https://www.kaggle.com/settings -> "Create New Token"
#      and save it as ~/.kaggle/kaggle.json (chmod 600), or export
#      KAGGLE_USERNAME and KAGGLE_KEY.
#   3. Open https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset once in
#      a browser so that you have accepted the dataset's terms.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$DEST"

if [[ -f "$DEST/events.csv" ]]; then
  echo "events.csv already present in $DEST: nothing to do."
  exit 0
fi

if ! command -v kaggle >/dev/null 2>&1; then
  echo "The kaggle CLI is not installed. Run: pip install kaggle" >&2
  exit 1
fi

kaggle datasets download -d retailrocket/ecommerce-dataset -p "$DEST" --unzip
echo
ls -lh "$DEST"
