"""

HotFlip
===========
(HotFlip: White-Box Adversarial Examples for Text Classification)

"""
from textattack import Attack
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import BeamSearch
from textattack.transformations import WordSwapGradientBased

from .attack_recipe import AttackRecipe


class HotFlipEbrahimi2017(AttackRecipe):
    """Ebrahimi, J. et al. (2017)

    HotFlip: White-Box Adversarial Examples for Text Classification

    https://arxiv.org/abs/1712.06751

    This is a reproduction of the HotFlip word-level attack (section 5 of the
    paper).
    """

    @staticmethod
    def build(model_wrapper):
        #
        # "HotFlip ... uses the gradient with respect to a one-hot input
        # representation to efficiently estimate which individual change has the
        # highest estimated loss."
        transformation = WordSwapGradientBased(model_wrapper, top_n=1)
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        # 0. "We were able to create only 41 examples (2% of the correctly-
        # classified instances of the SST test set) with one or two flips."
        #
        constraints.append(MaxWordsPerturbed(max_num_words=2))
        #
        # 1. "The cosine similarity between the embedding of words is bigger than a
        #   threshold (0.8)."
        #
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.8))
        #
        # 2. "The two words have the same part-of-speech."
        #
        constraints.append(PartOfSpeech())
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # "HotFlip ... uses a beam search to find a set of manipulations that work
        # well together to confuse a classifier ... The adversary uses a beam size
        # of 10."
        #
        search_method = BeamSearch(beam_width=10)

        return Attack(goal_function, constraints, transformation, search_method)
