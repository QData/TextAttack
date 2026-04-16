# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TextAttack (v0.3.10) is a Python framework for adversarial attacks, data augmentation, and model training in NLP. It provides a modular system where attacks are composed of four pluggable components: goal functions, constraints, transformations, and search methods. The project is maintained by UVA QData Lab.

## Common Commands

### Installation (dev mode)
```bash
pip install -e .[dev]
```

### Testing
```bash
make test                    # Run full test suite (pytest --dist=loadfile -n auto)
pytest tests -v              # Verbose test run
pytest tests/test_augment_api.py  # Run a single test file
pytest --lf                  # Re-run only last failed tests
```

### Formatting & Linting
```bash
make format    # Auto-format with black, isort, docformatter
make lint      # Check formatting (black --check, isort --check-only, flake8)
```

### Building Docs
```bash
make docs       # Build HTML docs with Sphinx
make docs-auto  # Hot-reload docs server on port 8765
```

### CLI Usage
```bash
textattack attack --recipe textfooler --model bert-base-uncased-mr --num-examples 100
textattack augment --input-csv examples.csv --output-csv output.csv --input-column text --recipe embedding
textattack train --model-name-or-path lstm --dataset yelp_polarity --epochs 50
textattack list attack-recipes
textattack peek-dataset --dataset-from-huggingface snli
```

## Architecture

### Core Attack Pipeline (`textattack/attack.py`, `textattack/attacker.py`)

An `Attack` is composed of exactly four components:
1. **GoalFunction** (`textattack/goal_functions/`) - Determines if an attack succeeded. Categories: `classification/` (untargeted, targeted), `text/` (BLEU, translation overlap), `custom/`.
2. **Constraints** (`textattack/constraints/`) - Filter invalid perturbations. Categories: `semantics/` (sentence encoders, word embeddings), `grammaticality/` (POS, language models, grammar tools), `overlap/` (edit distance, BLEU), `pre_transformation/` (restrict search space before transforming).
3. **Transformation** (`textattack/transformations/`) - Generate candidate perturbations. Types: `word_swaps/` (embedding, gradient, homoglyph, WordNet), `word_insertions/`, `word_merges/`, `sentence_transformations/`, `WordDeletion`, `CompositeTransformation`.
4. **SearchMethod** (`textattack/search_methods/`) - Traverse the perturbation space. Includes: `BeamSearch`, `GreedySearch`, `GreedyWordSwapWIR`, `GeneticAlgorithm`, `ParticleSwarmOptimization`, `DifferentialEvolution`.

The `Attacker` class orchestrates running attacks on datasets with parallel processing, checkpointing, and logging.

### Attack Recipes (`textattack/attack_recipes/`)

Pre-built attack configurations from the literature (e.g., TextFooler, DeepWordBug, BAE, BERT-Attack, CLARE, CheckList, etc.). Each recipe subclasses `AttackRecipe` and implements a `build(model_wrapper)` classmethod that returns a configured `Attack` object. Includes multi-lingual recipes for French, Spanish, and Chinese.

### Key Abstractions

- **`AttackedText`** (`textattack/shared/attacked_text.py`) - Central text representation that maintains both token list and original text with punctuation. Used throughout the pipeline instead of raw strings.
- **`ModelWrapper`** (`textattack/models/wrappers/`) - Abstract interface for models. Implementations for PyTorch, HuggingFace, TensorFlow, sklearn. Models must accept string input and return predictions.
- **`Dataset`** (`textattack/datasets/`) - Iterable of `(input, output)` pairs. Supports HuggingFace datasets and custom files.
- **`Augmenter`** (`textattack/augmentation/`) - Uses transformations and constraints for data augmentation (not adversarial attacks). Built-in recipes: wordnet, embedding, charswap, eda, checklist, clare, back_trans.
- **`PromptAugmentationPipeline`** (`textattack/prompt_augmentation/`) - Augments prompts and generates LLM responses.
- **LLM Wrappers** (`textattack/llms/`) - Wrappers for using LLMs (HuggingFace, ChatGPT) with prompt augmentation.

### CLI Commands (`textattack/commands/`)

Entry point: `textattack/commands/textattack_cli.py`. Each command (attack, augment, train, eval-model, list, peek-dataset, benchmark-recipe, attack-resume) is a subclass of `TextAttackCommand` with `register_subcommand()` and `run()` methods.

### Configuration

- Version tracked in `docs/conf.py` (imported by `setup.py`)
- Cache directory: `~/.cache/textattack/` (override with `TA_CACHE_DIR` env var)
- Formatting: black (line length 88), isort (skip `__init__.py`), flake8 (ignores: E203, E266, E501, W503, D203)

### CI Workflows (`.github/workflows/`)

- `check-formatting.yml` - Runs `make lint` on Python 3.9
- `run-pytest.yml` - Sets up Python 3.8/3.9 (pytest currently skipped in CI)
- `publish-to-pypi.yml` - PyPI publishing
- `make-docs.yml` - Documentation build
- `codeql-analysis.yml` - Security analysis

### Test Structure

Tests are in `tests/` organized by feature:
- `test_command_line/` - CLI command integration tests (attack, augment, train, eval, list, loggers)
- `test_constraints/` - Constraint unit tests
- `test_augment_api.py`, `test_transformations.py`, `test_attacked_text.py`, `test_tokenizers.py`, `test_word_embedding.py`, `test_metric_api.py`, `test_prompt_augmentation.py`
- `test_command_line/update_test_outputs.py` - Script to regenerate expected test outputs

### Adding New Components

- **Attack recipe**: Subclass `AttackRecipe` in `textattack/attack_recipes/`, implement `build(model_wrapper)`, add import to `__init__.py`, add doc reference in `docs/attack_recipes.rst`.
- **Transformation**: Subclass `Transformation` in appropriate subfolder under `textattack/transformations/`.
- **Constraint**: Subclass `Constraint` or `PreTransformationConstraint` in appropriate subfolder under `textattack/constraints/`.
- **Search method**: Subclass `SearchMethod` in `textattack/search_methods/`.
