# Changelog

All notable changes to this project will be documented in this file.

This file was started alongside the `leap` attack recipe addition below;
it does not attempt to reconstruct history prior to that point. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

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
