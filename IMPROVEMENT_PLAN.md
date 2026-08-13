# TextAttack Codebase Improvement Plan

A prioritized, holistic plan for modernizing and hardening the TextAttack codebase. Each item includes rationale, affected files, and suggested approach.

**Guiding principle:** Infrastructure, tooling, and non-functional improvements come first so that functional changes benefit from better CI, packaging, and code quality foundations.

---

## Priority 1 — Critical (Infrastructure & CI)

### 1.1 Re-enable tests in CI

**Why:** The pytest step in CI is completely commented out (`echo "skipping tests!"` in `run-pytest.yml` line 55). This means every merged PR bypasses the test suite. Without CI tests, regressions accumulate silently, and contributors have no automated safety net. This must be fixed first — all subsequent changes need CI to validate them.

**Affected file:** `.github/workflows/run-pytest.yml` (lines 54–56)

**Suggested approach:** Uncomment the `pytest tests -v` line. If tests are failing and that's why they were disabled, fix the failing tests first — disabling CI is not a sustainable workaround.

### 1.2 Update CI infrastructure

**Why:** All GitHub Actions workflows use `actions/checkout@v2` and `actions/setup-python@v2`, which are deprecated and will eventually stop working. The CodeQL workflow uses `v1` actions. The Python version matrix only covers 3.8 and 3.9 — Python 3.8 reached end-of-life in October 2024, and 3.10–3.12 are untested.

**Affected files:** All `.github/workflows/*.yml` files (5 files)

**Suggested approach:**
- Update to `actions/checkout@v4`, `actions/setup-python@v5`, `github/codeql-action/*@v3`.
- Expand Python matrix to `[3.9, 3.10, 3.11, 3.12]`.
- Drop 3.8 from the matrix and update `python_requires` in setup metadata.
- Replace `python setup.py sdist bdist_wheel` with `python -m build` in publish workflow.

### 1.3 Update pinned dev tool versions

**Why:** Test extras pin `black==20.8b1` (from August 2020) and `isort==5.6.4` (from 2020). These versions are incompatible with Python 3.10+ and miss years of bug fixes and formatting improvements. Contributors on modern Python cannot install the dev extras.

**Affected file:** `setup.py` (lines 20–27)

**Suggested approach:** Update to current stable versions (`black>=23.0`, `isort>=5.12`). Consider using `pre-commit` to manage formatting tool versions consistently across contributors.

---

## Priority 2 — High (Packaging & Dependencies)

### 2.1 Modernize packaging: add `pyproject.toml`

**Why:** The project relies solely on `setup.py`, which is the legacy packaging approach. PEP 517/518 (`pyproject.toml`) is now the standard. The current setup also has fragile patterns: version is imported from `docs/conf.py` at build time (cross-directory import that can break in isolated builds), and `requirements.txt` is read via `open().readlines()` without stripping whitespace.

**Affected files:** `setup.py`, `docs/conf.py`, `textattack/__init__.py`

**Suggested approach:**
- Create `pyproject.toml` with build-system metadata, dependencies, and project metadata.
- Move the version string to `textattack/__init__.py` as `__version__` (users expect `textattack.__version__` to work — it currently doesn't exist).
- Replace `setup.py` with a minimal shim or remove it entirely.

### 2.2 Fix dependency version constraints

**Why:** 15 of 22 runtime dependencies in `requirements.txt` have no version constraint at all (e.g., `flair`, `nltk`, `language_tool_python`). This means a new release of any of these can silently break TextAttack. The remaining dependencies use only `>=` lower bounds with no upper bounds, which provides minimal protection.

**Affected file:** `requirements.txt`

**Suggested approach:** Add compatible-release constraints (`~=`) or upper bounds for all dependencies. At minimum, pin major versions (e.g., `flair>=0.12,<1.0`). Run `pip freeze` on a known-good environment to establish baseline versions.

---

## Priority 3 — Medium (Non-Functional Code Quality)

### 3.1 Externalize the 10,669-line `data.py` file

**Why:** `textattack/shared/data.py` is a single 10,669-line file containing only hardcoded named entity lists (country names, person names, etc.). This makes git diffs noisy, IDE indexing slow, and the module hard to navigate. It inflates the package size unnecessarily as Python source.

**Affected file:** `textattack/shared/data.py`

**Suggested approach:** Move data into JSON or text files under a `textattack/data/` directory. Load them lazily at first use. This also makes it easier for users to customize or extend the lists.

### 3.2 Replace deprecated `logger.warn()` with `logger.warning()`

**Why:** `logger.warn()` has been deprecated since Python 3.2 and may be removed in a future version. It already emits deprecation warnings in some environments.

**Affected files:**
- `textattack/attacker.py` (lines 94, 182, 353)
- `textattack/trainer.py` (line 116)
- `textattack/shared/validators.py` (lines 59, 74, 83)
- `textattack/shared/utils/misc.py` (line 68)

**Suggested approach:** Global find-and-replace of `.warn(` with `.warning(` in these files.

### 3.3 Add type hints to core classes

**Why:** The core classes (`Attack`, `Attacker`, `GoalFunction`, `SearchMethod`) have essentially zero return type hints. This makes IDE autocompletion unreliable, prevents static analysis from catching bugs, and forces new contributors to read implementation to understand interfaces. `AttackedText` is partially typed (~80%) but inconsistent.

**Affected files:**
- `textattack/attack.py` — 16 methods, 0 return type hints
- `textattack/attacker.py` — 11 methods, 0 return type hints
- `textattack/goal_functions/goal_function.py` — 18 methods, 0 return type hints
- `textattack/search_methods/search_method.py` — abstract class, no return types

**Suggested approach:** Add return type annotations to all public methods in these four files first. Use `-> None`, `-> List[AttackedText]`, `-> AttackResult`, etc. This can be done incrementally without breaking changes.

### 3.4 Replace star imports with explicit imports

**Why:** Several `__init__.py` files use `from .module import *`, which makes it unclear what names are exported, can cause naming collisions, and breaks static analysis tools.

**Affected files:**
- `textattack/shared/utils/__init__.py` (lines 1–5)
- `textattack/goal_functions/__init__.py` (lines 11–13)
- `textattack/transformations/__init__.py` (lines 11–14)

**Suggested approach:** Replace star imports with explicit name lists. If maintaining `__all__` in submodules, that's acceptable — but the importing modules should still list names explicitly.

### 3.5 Clean up `.gitignore`

**Why:** The `.gitignore` contains a suspicious line `textattack/=22.3.0` (line 52) that looks like leftover state from pip output, not a valid ignore pattern.

**Suggested approach:** Remove the invalid line. Audit remaining entries for completeness (add `.env` if missing).

### 3.6 Add `tests/conftest.py` and expand test coverage

**Why:** There is no shared test infrastructure (`conftest.py`). Core classes `Attack`, `Attacker`, `GoalFunction`, and `SearchMethod` have no dedicated unit tests. There's a TODO in `test_attacked_text.py` for missing `align_words_with_tokens` tests.

**Suggested approach:** Create `tests/conftest.py` with shared fixtures (mock models, sample texts, etc.). Add unit tests for core classes. Prioritize testing the attack pipeline and search methods.

---

## Priority 4 — High (Functional Fixes — Security & Correctness)

These items change runtime behavior. They are ordered after infrastructure so that CI, packaging, and tests are in place to validate them.

### 4.1 Replace all `eval()` calls with a safe registry/factory pattern

**Why:** The codebase uses `eval()` extensively to instantiate components from user-supplied strings (attack recipes, transformations, goal functions, constraints, search methods). While inputs are partially validated against predefined dictionaries, `eval()` remains an inherent code-injection vector — especially dangerous in a library that accepts CLI arguments. Any future change that loosens the validation or introduces a new code path could expose users to arbitrary code execution.

**Affected files:**
- `textattack/attack_args.py` (lines 623–752) — transformations, goal functions, constraints, search methods, recipes
- `textattack/model_args.py` (line 285) — model class instantiation
- `textattack/dataset_args.py` (line 243) — dataset instantiation
- `textattack/training_args.py` (line 589) — attack recipe instantiation
- `textattack/commands/augment_command.py` (lines 36, 84, 182) — augmentation recipes

**Suggested approach:** Introduce a registry dict mapping string names to classes (e.g., `TRANSFORMATION_REGISTRY = {"word-swap-embedding": WordSwapEmbedding, ...}`). Use `getattr()` on known modules as a fallback. This is safer, faster, and easier to debug than `eval()`.

### 4.2 Fix the `update_attack_args()` bug

**Why:** This is a silent logic bug — the method appears to work but never actually updates the intended attribute. It always writes to a literal attribute named `k` instead of the dynamic key.

**Affected file:** `textattack/attacker.py` (line 460)

```python
# Current (broken):
self.attack_args.k = kwargs[k]

# Fix:
setattr(self.attack_args, k, kwargs[k])
```

**Why necessary:** Any code calling `attacker.update_attack_args(num_examples=100)` silently fails. This is a correctness bug that could cause wrong experimental results.

### 4.3 Replace `assert` with proper exceptions for input validation

**Why:** Python's `assert` statements are removed when running with `-O` (optimize) flag. Using them for input validation means all runtime checks silently vanish in optimized mode. This is particularly dangerous for a library where users may run in optimized mode for performance.

**Affected files:**
- `textattack/attack.py` (lines 93–108) — validates constructor arguments
- `textattack/attacker.py` (lines 70–80) — validates attack args
- `textattack/attack_args.py` (lines 230–246) — validates configuration

**Suggested approach:** Replace `assert condition, message` with `if not condition: raise TypeError(message)` or `ValueError(message)` as appropriate.

### 4.4 Fix error handling anti-patterns

**Why:** Several error handling patterns reduce debuggability and correctness:
- `except Exception as e: raise e` (attacker.py:170) — destroys the original traceback by re-raising via variable instead of bare `raise`
- `logging.disable()` without arguments (attacker.py:569) — globally disables ALL logging for the entire process, not just TextAttack
- `torch.cuda.empty_cache()` called without `torch.cuda.is_available()` guard — can fail on CPU-only systems

**Suggested approach:**
- Change `raise e` to `raise` to preserve traceback
- Replace `logging.disable()` with `logger.setLevel(logging.CRITICAL)` for module-scoped control
- Add `if torch.cuda.is_available():` guard before CUDA calls

### 4.5 Eliminate module-level side effects

**Why:** Several modules execute side effects at import time: downloading data, calling `torch.cuda.empty_cache()`, setting environment variables, and importing heavy optional dependencies. This slows down `import textattack`, causes failures when optional deps are missing, and makes testing difficult because imports are no longer pure.

**Affected files:**
- `textattack/shared/utils/install.py` (lines 203–210) — runs `_post_install_if_needed()` on import, which downloads NLTK data and does network I/O
- `textattack/shared/utils/strings.py` (lines 4–5) — top-level `import flair; import jieba` (should be lazy)
- `textattack/models/wrappers/huggingface_model_wrapper.py` (line 15) — `torch.cuda.empty_cache()` at module level
- `textattack/models/wrappers/pytorch_model_wrapper.py` (line 13) — same issue

**Suggested approach:** Defer all side effects to first use. Use the `LazyLoader` pattern (already present in the codebase) for optional dependencies. Move CUDA cache clearing into method bodies. Gate network downloads behind explicit function calls.

### 4.6 Fix thread-safety issue in prompt augmentation

**Why:** `textattack/prompt_augmentation/prompt_augmentation_pipeline.py` (lines 31–41) mutates a shared augmenter's constraint list by appending a constraint, running augmentation, then popping it off. If an exception occurs between the append and pop, the constraint list is left in a corrupted state. This is also not thread-safe.

**Suggested approach:** Create a copy of the constraints list or pass constraints as a parameter rather than mutating shared state.

### 4.7 Use safer serialization where possible

**Why:** Multiple files use `pickle.load()` to deserialize data downloaded from S3 or user-provided checkpoints. Pickle can execute arbitrary code during deserialization.

**Affected files:**
- `textattack/shared/checkpoint.py` (lines 221, 226)
- `textattack/shared/word_embeddings.py` (lines 296–298)
- `textattack/transformations/word_swaps/word_swap_hownet.py` (line 30)

**Suggested approach:** For internally-produced data (embeddings, candidate banks), migrate to safer formats (NumPy `.npy`, JSON, or `safetensors`). For checkpoints, add a warning in documentation about only loading trusted checkpoints. This is a longer-term migration.

---

## Priority 5 — Low (New Features & Long-term Debt)

### 5.1 Expand LLM integration

**Why:** The `textattack/llms/` module contains only two thin wrappers (`ChatGPTWrapper`, `HuggingFaceLLMWrapper`). The ChatGPT wrapper has no retry logic, timeout handling, rate limiting, or error handling for missing API keys. These wrappers are not integrated into the main attack pipeline or documented.

**Suggested approach:** Add proper error handling and retry logic to existing wrappers. Integrate LLM wrappers into the model wrapper hierarchy so they can be used with existing attacks. Document usage in the README and examples.

### 5.2 Resolve accumulated TODOs

**Why:** There are 13+ TODO/FIXME/HACK comments scattered across the codebase representing unresolved technical debt. Some are non-trivial bugs:
- `trainer.py:227` — TODO about ground truth manipulation bug
- `particle_swarm_optimization.py:67` — TODO about slow memory buildup
- `word_embedding_distance.py:69` — FIXME: index sometimes larger than tokens-1
- `attacked_text.py:460` — TODO about undefined punctuation behavior

**Suggested approach:** Triage each TODO into a GitHub issue with severity label. Fix the bug-class TODOs (trainer, PSO memory, embedding index) as part of Priority 4 work. Convert informational TODOs into GitHub issues and remove the comments.

---

## Summary

| Priority | Items | Theme |
|----------|-------|-------|
| **P1 — Critical** | 1.1–1.3 | CI re-enablement, CI modernization, dev tooling |
| **P2 — High** | 2.1–2.2 | Packaging modernization, dependency safety |
| **P3 — Medium** | 3.1–3.6 | Non-functional code quality, type hints, tests |
| **P4 — High** | 4.1–4.7 | Functional fixes: security, correctness, runtime behavior |
| **P5 — Low** | 5.1–5.2 | New features, tech debt cleanup |

**Recommended execution order:** Start with P1 (CI & tooling) so all subsequent changes are validated automatically. Then P2 (packaging & deps) to stabilize the build. P3 (non-functional quality) can proceed in parallel. P4 (functional changes) comes after CI and tests are solid, ensuring behavioral changes are well-tested. P5 items are opportunistic or good first-contributor issues.
