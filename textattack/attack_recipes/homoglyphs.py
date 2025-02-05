"""

TextFooler (Is BERT Really Robust?)
===================================================
A Strong Baseline for Natural Language Attack on Text Classification and Entailment)

"""

from textattack import Attack
from textattack.goal_functions import LevenshteinExceedsTargetDistance
from textattack.search_methods import DifferentialEvolutionSearch
from textattack.transformations import WordSwapHomoglyphSwap

from .attack_recipe import AttackRecipe


class Homoglyphs(AttackRecipe):
    """Jin, D., Jin, Z., Zhou, J.T., & Szolovits, P. (2019).

    Is BERT Really Robust? Natural Language Attack on Text
    Classification and Entailment.

    https://arxiv.org/abs/1907.11932
    """

    @staticmethod
    def build(model_wrapper):
        transformation = WordSwapHomoglyphSwap()
        constraints = []
        goal_function = LevenshteinExceedsTargetDistance(model_wrapper)
        search_method = DifferentialEvolutionSearch(popsize = 5, maxiter = 5, verbose = True)

        return Attack(goal_function, constraints, transformation, search_method)
