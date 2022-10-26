"""

Seq2Sick
================================================
(Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples)
"""
from eukaryote import Attack
from eukaryote.constraints.overlap import LevenshteinEditDistance
from eukaryote.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from eukaryote.goal_functions import NonOverlappingOutput
from eukaryote.search_methods import GreedyWordSwapWIR
from eukaryote.transformations import WordSwapEmbedding

from .attack_recipe import AttackRecipe


class Seq2SickCheng2018BlackBox(AttackRecipe):
    """Cheng, Minhao, et al.

    Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with
    Adversarial Examples

    https://arxiv.org/abs/1803.01128

    This is a greedy re-implementation of the seq2sick attack method. It does
    not use gradient descent.
    """

    @staticmethod
    def build(model_wrapper, goal_function="non_overlapping"):

        #
        # Goal is non-overlapping output.
        #
        goal_function = NonOverlappingOutput(model_wrapper)
        transformation = WordSwapEmbedding(max_candidates=50)
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        # In these experiments, we hold the maximum difference
        # on edit distance (ϵ) to a constant 30 for each sample.
        #
        constraints.append(LevenshteinEditDistance(30))
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = GreedyWordSwapWIR(wir_method="unk")

        return Attack(goal_function, constraints, transformation, search_method)
