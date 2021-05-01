"""

PWWS
=======

(Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency)

"""
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.shared.attack import Attack
from textattack.transformations import WordSwapWordNet

from .attack_recipe import AttackRecipe


class PWWSRen2019(AttackRecipe):
    """An implementation of Probability Weighted Word Saliency from "Generating
    Natural Langauge Adversarial Examples through Probability Weighted Word
    Saliency", Ren et al., 2019.

    Words are prioritized for a synonym-swap transformation based on
    a combination of their saliency score and maximum word-swap effectiveness.
    Note that this implementation does not include the Named
    Entity adversarial swap from the original paper, because it requires
    access to the full dataset and ground truth labels in advance.

    https://www.aclweb.org/anthology/P19-1103/
    """

    @staticmethod
    def build(model):
        transformation = WordSwapWordNet()
        constraints = [RepeatModification(), StopwordModification()]
        goal_function = UntargetedClassification(model)
        # search over words based on a combination of their saliency score, and how efficient the WordSwap transform is
        search_method = GreedyWordSwapWIR("weighted-saliency")
        return Attack(goal_function, constraints, transformation, search_method)
