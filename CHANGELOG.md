# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.3.11] - 2026-08-14

### Added

- `RemoteModelWrapper` (`textattack.models.wrappers.RemoteModelWrapper`): query a model served behind a remote HTTP API instead of running it locally. Request/response handling is adaptable to different endpoint schemas via `request_fn`/`response_fn`.
- New attack recipe `leap` (`LEAP2023`, see `textattack/attack_recipes/leap_2023.py`)
  implementing LEAP: Efficient and Automated Test Method for NLP Software
  ([arXiv:2308.11284](https://arxiv.org/abs/2308.11284)). LEAP is a
  Levy-flight/adaptive-inertia variant of the Particle Swarm Optimization
  search already used by the `pso` recipe (`PSOZang2020`); see
  `textattack/search_methods/particle_swarm_optimization_leap.py`
  (`ParticleSwarmOptimizationLEAP`, subclassing `ParticleSwarmOptimization`)
  for the algorithmic relationship between the two.
- `tests/test_attack_recipes.py`: structural and functional tests comparing
  `LEAP2023` against `PSOZang2020`, its closest existing recipe.
- `tests/benchmark_leap_vs_pso.py`: a manual (not CI-run) benchmark script
  comparing `LEAP2023` against a WordNet-transformation variant of vanilla
  `ParticleSwarmOptimization`, isolating the search algorithm from the
  candidate-word transformation. Results (`cnn-ag-news`, AG News test set)
  are documented in the "Benchmark" section of `LEAP2023`'s docstring:
  under an unrestricted query budget, both hit ~95% success (n=20), with
  LEAP using ~6% fewer queries and running ~2.25x faster; under a 2000-query
  budget (n=100), success collapses to ~23-24% for both, with LEAP still
  modestly ahead on every metric but by a much smaller margin.
- `train` command: `--dataset-from-file` support, matching what `attack`/`eval`
  already had. Point it at a Python module exposing `train_dataset`/
  `eval_dataset` (each a `textattack.datasets.Dataset`), optionally
  `path.py^prefix` for `prefix_train_dataset`/`prefix_eval_dataset` ([#625](https://github.com/QData/TextAttack/issues/625)).
- `HuggingFaceModelWrapper`: an optional `max_length` constructor argument,
  forwarded to `.generate()` for raw `transformers` encoder-decoder
  generation models. Left unset by default so a checkpoint's own
  `generation_config` isn't overridden.
- Docs: the single-example `Attack.attack(text, label)` API (already existed,
  wasn't documented) ([#673](https://github.com/QData/TextAttack/issues/673)); a working example attacking a raw
  `transformers.BartForConditionalGeneration`/`T5ForConditionalGeneration`
  via `Seq2SickCheng2018BlackBox` ([#772](https://github.com/QData/TextAttack/issues/772)); a "Multi-lingual attacks" section
  listing the French/Spanish/Chinese recipes ([#423](https://github.com/QData/TextAttack/issues/423)).

### Changed

- Refactored `ParticleSwarmOptimization.perform_search` (shared base class)
  to expose its per-iteration deltas as overridable hook methods
  (`_initialize_velocities`, `_pre_iteration_setup`, `_compute_omega`,
  `_compute_turn_prob`, `_compute_change_ratio`), matching the hook pattern
  `GeneticAlgorithm`/`AlzantotGeneticAlgorithm` already use elsewhere in this
  codebase. `ParticleSwarmOptimizationLEAP` now overrides only these hooks
  and `_perturb`, instead of duplicating a ~155-line copy of
  `perform_search`; the standalone `_greedy_perturb` method was folded into
  a `_perturb` override, so LEAP's mutation step can no longer silently
  fall back to the parent's probabilistic mutation by mistake.
- Replaced LEAP's hand-rolled `softmax` with `scipy.special.softmax`, and
  cached (`functools.lru_cache`) the alpha-invariant constants in its
  Levy-flight sampler (`sigmax`/`K`/`C`), since `alpha` is always `1.5` in
  this module -- both were previously recomputed from scratch on every call.
  The Levy-flight sampling algorithm itself (Mantegna's method, matching the
  authors' reference implementation) was intentionally left as-is rather
  than swapped for `scipy.stats.levy_stable`, since that would change the
  search's statistical behavior, not just its implementation.
- Renamed the `gamma` parameter of `levy()` to `scale`, since it shadowed
  the module-level `from scipy.special import gamma as gamma` import.

### Fixed

Several correctness issues found while porting LEAP against its authors'
reference implementation, some of which also affect the pre-existing `pso`
recipe since `ParticleSwarmOptimizationLEAP` shares code with
`ParticleSwarmOptimization`:

- LEAP's mutation step now calls its greedy mutation instead of silently
  falling back to the inherited, probabilistic `_perturb`, and computes
  `change_ratio` against each particle's own local elite instead of the
  original input, matching the reference implementation.
- Guarded LEAP's per-iteration adaptive inertia-weight interpolation against
  a zero/negative denominator (`fit_ave`/`fit_min` are frozen at the initial
  population's statistics, so a particle's score can drift below `fit_min`
  in later iterations).
- Fixed `ParticleSwarmOptimization.perform_search` (shared base class, also
  used by `pso`) letting `global_elite`/`local_elites` alias the same
  `PopulationMember` objects held in `population` -- both
  `global_elite = max(population, ...)` and `local_elites = copy.copy(population)`
  only copied the list, not its elements, so a particle never reassigned to
  a new object by `_turn` during an iteration (a real possibility whenever
  neither of that iteration's two random turn-probability checks fire)
  stayed aliased to its tracked elite. A later in-place mutation
  (`_perturb`) would then silently corrupt that elite. Every population
  member and both elites are now copied individually at initialization,
  in addition to the existing fix on `_turn`'s constraint-failure return
  path.
- Reset LEAP's per-iteration `omega` (inertia weight) list at the start of
  each iteration instead of accumulating it across the whole search; it was
  being indexed by particle position, so every iteration after the first
  was silently reading back iteration-0's stale values.
- Capped the retry count in LEAP's Levy-flight rejection sampling
  (`get_one_levy` / velocity initialization), which previously used an
  unbounded `while True` loop with no fallback.
- Fixed `docs/api/search_methods.rst`'s `ParticleSwarmOptimizationLEAP`
  section heading underline, which was shorter than the title.

A round of fixes for long-standing issues found during an issue-triage pass:

- `BERTAttackLi2020`: default `max_candidates` `48` -> `8`, since `48`
  makes the masked-LM candidate search explode combinatorially on
  multi-subword tokens, causing multi-hour runtimes ([#586](https://github.com/QData/TextAttack/issues/586)).
- `AttackedText.generate_new_attacked_text`: fixed corruption of the
  `<SPLIT>` join token when a replaced word (e.g. "I") is itself a
  character substring of `"<SPLIT>"` and directly follows it in the
  joined text ([#631](https://github.com/QData/TextAttack/issues/631)).
- `WordSwapInflections`: restored matching against flair's current
  `"upos-fast"` tags (`NOUN`/`VERB`/`ADJ`/`PROPN`/`AUX`), which
  `AttackedText.pos_of_word_index` had switched to emitting directly a
  while back, leaving the transformation silently returning zero
  candidates for ordinary words ([#713](https://github.com/QData/TextAttack/issues/713), [#727](https://github.com/QData/TextAttack/issues/727)).
- `textattack.shared.utils.flair_tag`: cached one tagger per `tag_type`
  instead of a single global slot, which silently reused whichever
  tagger (POS or NER) loaded first for every later call regardless of
  the requested `tag_type` -- corrupting POS/NER results for whichever
  came second in the same process.
- `GreedyWordSwapWIR`: actually implemented `truncate_words_to` (a2t's
  recipe had been passing it since an earlier PR, but the constructor
  never accepted it, crashing on init) ([#754](https://github.com/QData/TextAttack/issues/754)); for `wir_method="gradient"`,
  the truncation now also bounds the expensive `get_grad` call itself
  (not just the cheap post-hoc index-scoring loop), and sorts
  `indices_to_order` first since it can arrive in non-ascending order
  from a `set`-derived source.
- `textattack/shared/validators.py`: the model-compatibility regex for
  classification models only matched the pre-4.x
  `transformers.modeling_<model>` layout; now also matches
  `transformers.models.<model>.modeling_<model>` ([#722](https://github.com/QData/TextAttack/issues/722)). Also added a
  matching entry for raw `transformers` encoder-decoder generation
  classes (`T5ForConditionalGeneration`, `BartForConditionalGeneration`,
  ...), not just TextAttack's own `T5ForTextToText` helper, which used
  to print a spurious compatibility warning for every attack against one
  ([#771](https://github.com/QData/TextAttack/issues/771)).
- `AttackArgs`: warn (rather than silently drop) when `num_examples` is
  explicitly set alongside `num_successful_examples`, since the latter
  overrides the former and users combining both had no way to tell why
  `num_examples` came back `None` ([#728](https://github.com/QData/TextAttack/issues/728)).
- `AttackedText.words_diff_ratio`: comparing two Python lists with `!=`
  yields a single bool, not an elementwise mask, so this always returned
  `0` or `1` regardless of how many words actually differed; fixed by
  comparing as numpy arrays ([#787](https://github.com/QData/TextAttack/issues/787)).
- `Augmenter.augment`: bound-retry the outer sampling loop so a
  transformation with limited output diversity for a given input (e.g.
  `BackTranslationAugmenter`'s random language chaining colliding on
  short sentences) doesn't silently return fewer than
  `transformations_per_example` unique augmentations ([#800](https://github.com/QData/TextAttack/issues/800)); a later
  version of that same fix made `high_yield=True` mode plateau at roughly
  half its previous output instead of scaling with
  `transformations_per_example` (a single outer pass can add several
  results at once in that mode), and its final downsampling step called
  `random.sample()` on a `set`, which Python 3.11+ no longer accepts.
- `words_from_text`: strip allowed marks (quotes/hyphens/etc.) from both
  ends of a word, not just the leading end, so a quoted word like
  `"'CCC'"` doesn't keep its trailing quote ([#723](https://github.com/QData/TextAttack/issues/723)).
- `HuggingFaceModelWrapper`: route encoder-decoder generation models
  (e.g. a raw `BartForConditionalGeneration` loaded directly from
  `transformers`) through `.generate()` + decode instead of a plain
  forward pass, which only returns logits and breaks text-to-text goal
  functions expecting strings ([#771](https://github.com/QData/TextAttack/issues/771)). Routing prefers `model.can_generate()`
  over `hasattr(model, "generate")`, since on `transformers` versions
  predating `can_generate()`, every `PreTrainedModel` exposed `.generate`
  regardless of whether it had a generation-capable head, risking
  misrouting a seq2seq-backbone classification model.
- `Attack.cuda_`/`cpu_`: skip re-placing a `transformers.PreTrainedModel`
  that already has an `hf_device_map` (i.e. was loaded with
  `device_map=...` across multiple GPUs via `accelerate`), since forcing
  it onto a single device breaks that placement ([#798](https://github.com/QData/TextAttack/issues/798)); `cpu_` was missing
  this guard entirely, and the guard is now scoped to
  `transformers.PreTrainedModel` specifically rather than any
  `torch.nn.Module`, since this visitor also traverses non-HuggingFace
  models reachable from a `Constraint`/`GoalFunction`/`Transformation`.
- `Trainer.training_step`/`evaluate_step`: pad to the longest sequence in
  the batch (`padding=True`) instead of the tokenizer's static max
  length, avoiding wasted compute on shorter batches ([#737](https://github.com/QData/TextAttack/issues/737)).
