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
from textattack.search_methods import LEAPParticleSwarmOptimization
from textattack.transformations import WordSwapWordNet

from .attack_recipe import AttackRecipe

class LEAP2023(AttackRecipe):
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
        search_method = LEAPParticleSwarmOptimization(pop_size=60, max_iters=20, post_turn_check=True, max_turn_retries=20)

        return Attack(goal_function, constraints, transformation, search_method)
