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

### Fixed

Several correctness issues found while porting LEAP against its authors'
reference implementation, some of which also affect the pre-existing `pso`
recipe since `ParticleSwarmOptimizationLEAP` shares code with
`ParticleSwarmOptimization`:

- LEAP's mutation step now calls its greedy mutation (`_greedy_perturb`)
  instead of silently falling back to the inherited, probabilistic
  `_perturb`, and computes `change_ratio` against each particle's own local
  elite instead of the original input, matching the reference implementation.
- Guarded LEAP's per-iteration adaptive inertia-weight interpolation against
  a zero/negative denominator (`fit_ave`/`fit_min` are frozen at the initial
  population's statistics, so a particle's score can drift below `fit_min`
  in later iterations).
- Fixed `ParticleSwarmOptimization._turn` (shared base class, also used by
  `pso`) returning an elite `PopulationMember` by reference instead of a
  copy when no constraint-passing move was found, which could let a later
  in-place mutation silently corrupt a tracked local/global elite.
- Reset LEAP's per-iteration `omega` (inertia weight) list at the start of
  each iteration instead of accumulating it across the whole search; it was
  being indexed by particle position, so every iteration after the first
  was silently reading back iteration-0's stale values.
- Capped the retry count in LEAP's Levy-flight rejection sampling
  (`get_one_levy` / velocity initialization), which previously used an
  unbounded `while True` loop with no fallback.
- Fixed `docs/api/search_methods.rst`'s `ParticleSwarmOptimizationLEAP`
  section heading underline, which was shorter than the title.
