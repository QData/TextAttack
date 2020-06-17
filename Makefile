format: FORCE  ## Run black and isort (rewriting files)
	black .
	isort  --atomic --recursive tests textattack


lint: FORCE  ## Run black (in check mode)
	black . --check
	isort --check-only --recursive tests textattack

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
