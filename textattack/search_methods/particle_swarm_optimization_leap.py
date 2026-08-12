"""

LEAP Particle Swarm Optimization
====================================

LEAP, an automated test method that uses LEvy flight-based Adaptive Particle
swarm optimization integrated with textual features to generate adversarial test cases.

Subclasses :class:`~textattack.search_methods.ParticleSwarmOptimization`
(the reimplementation of Zang et al.'s PSO search used by
:class:`~textattack.attack_recipes.PSOZang2020`) via the parent class's
hook methods, overriding population-velocity initialization (Levy-flight
sampled), the per-particle inertia weight (:meth:`_compute_omega`), the
turn-probability normalization (:meth:`_compute_turn_prob`), the mutation
change-ratio reference point (:meth:`_compute_change_ratio`), and the
mutation step itself (:meth:`_perturb`, greedy instead of the parent's
probabilistic sampling) to more directly balance exploration and
exploitation across iterations. ``perform_search`` itself is inherited
unchanged.

`<https://arxiv.org/abs/2308.11284>`_
`<https://github.com/lumos-xiao/LEAP>`_
"""

from functools import lru_cache

import numpy as np
from scipy.special import gamma as gamma
from scipy.special import softmax

from textattack.search_methods import ParticleSwarmOptimization

# alpha is hardcoded to 1.5 everywhere this module calls `levy`/`get_one_levy`,
# so the alpha-dependent constants below (each a handful of `scipy.special.gamma`
# evaluations) are cached rather than recomputed on every one of the up to
# `pop_size * max_iters` calls into this sampler during a single search.


@lru_cache(maxsize=None)
def sigmax(alpha):
    numerator = gamma(alpha + 1.0) * np.sin(np.pi * alpha / 2.0)
    denominator = gamma((alpha + 1) / 2.0) * alpha * np.power(2.0, (alpha - 1.0) / 2.0)
    return np.power(numerator / denominator, 1.0 / alpha)


def vf(alpha):
    x = np.random.normal(0, 1)
    y = np.random.normal(0, 1)

    x = x * sigmax(alpha)

    return x / np.power(np.abs(y), 1.0 / alpha)


@lru_cache(maxsize=None)
def K(alpha):
    k = alpha * gamma((alpha + 1.0) / (2.0 * alpha)) / gamma(1.0 / alpha)
    k *= np.power(
        alpha
        * gamma((alpha + 1.0) / 2.0)
        / (gamma(alpha + 1.0) * np.sin(np.pi * alpha / 2.0)),
        1.0 / alpha,
    )

    return k


@lru_cache(maxsize=None)
def C(alpha):
    x = np.array(
        (0.75, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.95, 1.99)
    )
    y = np.array(
        (
            2.2085,
            2.483,
            2.7675,
            2.945,
            2.941,
            2.9005,
            2.8315,
            2.737,
            2.6125,
            2.4465,
            2.206,
            1.7915,
            1.3925,
            0.6089,
        )
    )

    return np.interp(alpha, x, y)


def levy(alpha, scale=1, n=1):
    w = 0
    for i in range(0, n):
        v = vf(alpha)

        while v < -10:
            v = vf(alpha)

        w += v * ((K(alpha) - 1.0) * np.exp(-v / C(alpha)) + 1.0)

    z = 1.0 / np.power(n, 1.0 / alpha) * w * scale

    return z


def get_one_levy(min, max, max_tries=1000):
    for _ in range(max_tries):
        temp = levy(1.5, 1)
        if min <= temp <= max:
            return temp
    # Exceedingly unlikely with a [0.5, 0.8]/[-v_max, v_max]-sized band, but
    # avoid an unbounded retry loop on an unlucky streak from the heavy-tailed
    # Levy distribution: fall back to a uniform draw within the target range.
    return np.random.uniform(min, max)


class ParticleSwarmOptimizationLEAP(ParticleSwarmOptimization):
    """Attacks a model with word substitutions using LEAP, a Levy-flight and
    per-particle-adaptive-inertia variant of the Particle Swarm Optimization
    (PSO) algorithm implemented by the parent class
    :class:`~textattack.search_methods.ParticleSwarmOptimization` (used by
    :class:`~textattack.attack_recipes.PSOZang2020`). See the module-level
    docstring above for what specifically differs from the parent class."""

    def _perturb(self, pop_member, original_result):
        """LEAP's mutation step: replace `pop_member` with the single best
        neighbor found across all word positions (greedy), rather than the
        parent's probabilistic sample among per-word best neighbors."""
        best_neighbors, prob_list = self._get_best_neighbors(
            pop_member.result, original_result
        )
        random_result = best_neighbors[np.argsort(prob_list)[-1]]

        if random_result == pop_member.result:
            return False
        else:
            pop_member.attacked_text = random_result.attacked_text
            pop_member.result = random_result
            return True

    def _initialize_velocities(self, num_words):
        v_init = np.empty(self.pop_size)
        v_init_rand = np.random.uniform(-self.v_max, self.v_max, self.pop_size)
        for i in range(self.pop_size):
            if np.random.uniform(-self.v_max, self.v_max) < levy(1.5, 1):
                v_init[i] = v_init_rand[i]
            else:
                # Only sample the (comparatively expensive) Levy draw when
                # the coin flip above actually selects this branch.
                v_init[i] = get_one_levy(-self.v_max, self.v_max)
        return np.array(
            [[v_init[t] for _ in range(num_words)] for t in range(self.pop_size)]
        )

    def _pre_iteration_setup(self, population):
        pop_fit = np.array([p.score for p in population])
        self._fit_ave = round(pop_fit.mean(), 3)
        self._fit_min = pop_fit.min()

    def _compute_omega(self, i, population):
        # `self._fit_ave`/`self._fit_min` are fixed at the initial population's
        # statistics (see `_pre_iteration_setup`), so a population member's
        # score can drift below `self._fit_min` in later iterations without
        # `self._fit_ave > self._fit_min` strictly holding; guard the
        # interpolation below against a zero (or negative) denominator.
        omega = np.empty(len(population))
        for k in range(len(population)):
            if population[k].score < self._fit_ave and self._fit_ave > self._fit_min:
                omega[k] = self.omega_2 + (
                    (population[k].score - self._fit_min)
                    * (self.omega_1 - self.omega_2)
                ) / (self._fit_ave - self._fit_min)
            else:
                omega[k] = get_one_levy(0.5, 0.8)
        return omega

    def _compute_turn_prob(self, velocities_k):
        # Unlike the parent class (which uses an independent per-word
        # sigmoid), LEAP normalizes turn probabilities across the whole
        # sentence with softmax, so on average ~1 word turns per call.
        # This matches the authors' reference implementation and is
        # intentional, not a drop-in-compatible substitute for sigmoid.
        return softmax(np.array([velocities_k]), axis=1)[0]

    def _compute_change_ratio(self, pop_member, local_elite, initial_result):
        return pop_member.attacked_text.words_diff_ratio(local_elite.attacked_text)
