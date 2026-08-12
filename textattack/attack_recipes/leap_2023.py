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
