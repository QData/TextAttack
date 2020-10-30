PEP_IGNORE_ERRORS="C901 E501 W503 E203 E231 E266 F403"

format: FORCE  ## Run black and isort (rewriting files)
	black .
	isort --atomic tests textattack
	docformatter --in-place --recursive textattack tests

lint: FORCE  ## Run black, isort, flake8 (in check mode)
	black . --check
	isort --check-only tests textattack
	flake8 . --count --ignore=$(PEP_IGNORE_ERRORS) --show-source --statistics --exclude=./.*,build,dist

test: FORCE ## Run tests using pytest
	python -m pytest --dist=loadfile -n auto

docs: FORCE ## Build docs using Sphinx.
	sphinx-build -b html docs docs/_build/html

docs-check: FORCE ## Builds docs using Sphinx. If there is an error, exit with an error code (instead of warning & continuing).
	sphinx-build -b html docs docs/_build/html -W

docs-auto: FORCE ## Build docs using Sphinx and run hotreload server using Sphinx autobuild.
	sphinx-autobuild docs docs/_build/html -H 0.0.0.0 -p 8765

all: format lint docs-check test ## Format, lint, and test. 

.PHONY: help

.DEFAULT_GOAL := help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

FORCE:
