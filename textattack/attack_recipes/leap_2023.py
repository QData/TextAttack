"""

LEAP
==================================

(LEAP: Efficient and Automated Test Method for NLP Software)

"""
from textattack import Attack
from textattack.constraints.pre_transformation import (
    MaxModificationRate,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import ParticleSwarmOptimizationLEAP
from textattack.transformations import WordSwapWordNet

from .attack_recipe import AttackRecipe


class LEAP2023(AttackRecipe):
    """LEAP: Efficient and Automated Test Method for NLP Software.

    https://arxiv.org/abs/2308.11284

    LEAP is a Levy-flight/adaptive-inertia variant of the Particle Swarm
    Optimization search used by :class:`~textattack.attack_recipes.PSOZang2020`
    (see :class:`~textattack.search_methods.ParticleSwarmOptimizationLEAP`,
    which subclasses :class:`~textattack.search_methods.ParticleSwarmOptimization`).
    Where PSOZang2020 swaps HowNet sememes and uses a fixed, globally-shared
    inertia weight, LEAP swaps WordNet synonyms, caps the total fraction of
    words modified via `MaxModificationRate`, and computes a per-particle
    inertia weight (Levy-flight sampled for above-average particles, fitness-
    interpolated otherwise) intended to better balance exploration and
    exploitation than PSOZang2020's shared decaying weight.

    Benchmark vs. PSOZang2020's search method (2026-08-12)
    --------------------------------------------------------
    Comparing search methods directly requires holding everything else
    fixed, since PSOZang2020 normally swaps HowNet sememes rather than
    WordNet synonyms -- see ``tests/benchmark_leap_vs_pso.py`` for the
    script (not run in CI) that reproduces this: a vanilla
    ``ParticleSwarmOptimization`` attack built with LEAP2023's own
    transformation/constraints/hyperparameters, so only the search method's
    internals differ. Run against ``cnn-ag-news`` on the AG News test set
    (first N examples, seed 765):

    +--------------------------+-----------------+--------------------+
    |                          | Unrestricted    | 2000-query budget  |
    |                          | budget (n=20)   | (n=100)            |
    +==========================+=================+====================+
    | LEAP success rate        | 95.0%           | 24.0%              |
    | PSO(WordNet) success rate| 95.0%           | 23.0%              |
    +--------------------------+-----------------+--------------------+
    | LEAP avg. queries        | 44,071.8        | 1,540.8            |
    | PSO(WordNet) avg. queries| 46,753.4        | 1,560.1            |
    +--------------------------+-----------------+--------------------+
    | LEAP avg. % words changed| 9.28%           | 4.51%              |
    | PSO(WordNet) avg. changed| 9.87%           | 5.04%              |
    +--------------------------+-----------------+--------------------+
    | LEAP wall time           | 157.3s          | 43.3s              |
    | PSO(WordNet) wall time   | 353.6s          | 46.5s              |
    +--------------------------+-----------------+--------------------+

    LEAP matches or modestly beats this apples-to-apples PSO baseline on
    every metric (success rate, query count, perturbation size, wall time),
    but the margin is small, not dramatic -- and it narrows further under a
    realistic query budget, where both methods' success rate collapses well
    below their unrestricted-budget numbers since neither search gets far
    enough in to fully exploit its strategy. Caveats: single model/dataset,
    a specific random seed, and this size of sample -- treat as directional,
    not definitive, evidence for LEAP's claimed advantage over vanilla PSO.
    """

    @staticmethod
    def build(model_wrapper):
        #
        # Swap words with their synonyms extracted based on the WordNet.
        #
        transformation = WordSwapWordNet()
        #
        # MaxModificationRate = 0.16 in AG's News
        #
        constraints = [MaxModificationRate(max_rate=0.16), StopwordModification()]
        #
        #
        # Use untargeted classification for demo, can be switched to targeted one
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Perform word substitution with LEAP algorithm.
        #
        search_method = ParticleSwarmOptimizationLEAP(
            pop_size=60, max_iters=20, post_turn_check=True, max_turn_retries=20
        )

        return Attack(goal_function, constraints, transformation, search_method)
