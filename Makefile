.PHONY: setup data test benchmark benchmark-synthetic notebooks notebooks-save clean-notebooks

PYTHON ?= python

setup:            ## install the package, notebook and dev dependencies
	$(PYTHON) -m pip install -e ".[dev,notebooks,data]"

data:             ## download RetailRocket from Kaggle into data/raw (needs a Kaggle API token)
	bash scripts/download_data.sh

test:             ## run the unit tests (synthetic data, ~10s)
	$(PYTHON) -m pytest

benchmark:        ## fit and evaluate every model (real data if present)
	$(PYTHON) scripts/run_benchmark.py

benchmark-synthetic:
	$(PYTHON) scripts/run_benchmark.py --synthetic

notebooks:        ## execute every notebook headlessly (smoke test; nothing is written)
	$(PYTHON) scripts/execute_notebooks.py

notebooks-save:   ## execute every notebook and save the outputs into it (e.g. after `make data`)
	$(PYTHON) scripts/execute_notebooks.py --save

clean-notebooks:  ## optional: strip all outputs in place
	$(PYTHON) scripts/execute_notebooks.py --clear-only
