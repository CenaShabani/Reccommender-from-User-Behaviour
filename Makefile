.PHONY: setup data test benchmark benchmark-synthetic notebooks clean-notebooks

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

notebooks:        ## execute every notebook headlessly (smoke test; outputs are not saved)
	$(PYTHON) scripts/execute_notebooks.py

clean-notebooks:  ## strip outputs before committing
	$(PYTHON) scripts/execute_notebooks.py --clear-only
