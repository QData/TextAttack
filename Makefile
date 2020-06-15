format: FORCE  ## Run isort and black (rewriting files)
	 isort -rc --atomic .
	 black .

lint: FORCE  ## Run flake8, mypy and black (in check mode)
	 black . --check

test: FORCE ## Run tests using pytest
	python -m pytest

docs: FORCE ## Build docs using Sphinx.
	sphinx-build -b html docs docs/_build/html 

docs-auto: FORCE ## Build docs using Sphinx and run hotreload server using Sphinx autobuild.
	sphinx-autobuild docs docs/_build/html -H 0.0.0.0 -p 8765

all: format lint test ## Format, lint, and test. 

.PHONY: help

.DEFAULT_GOAL := help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

FORCE:

