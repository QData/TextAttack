"""

Improved Genetic Algorithm
=============================

(Natural Language Adversarial Attacks and Defenses in Word Level)

"""
from eukaryote import Attack
from eukaryote.constraints.overlap import MaxWordsPerturbed
from eukaryote.constraints.pre_transformation import StopwordModification
from eukaryote.constraints.semantics import WordEmbeddingDistance
from eukaryote.goal_functions import UntargetedClassification
from eukaryote.search_methods import ImprovedGeneticAlgorithm
from eukaryote.transformations import WordSwapEmbedding

from .attack_recipe import AttackRecipe


class IGAWang2019(AttackRecipe):
    """Xiaosen Wang, Hao Jin, Kun He (2019).

    Natural Language Adversarial Attack and Defense in Word Level.

    http://arxiv.org/abs/1909.06723
    """

    @staticmethod
    def build(model_wrapper):
        #
        # Swap words with their embedding nearest-neighbors.
        # Embedding: Counter-fitted Paragram Embeddings.
        # Fix the hyperparameter value to N = Unrestricted (50)."
        #
        transformation = WordSwapEmbedding(max_candidates=50)
        #
        # Don't modify the stopwords
        #
        constraints = [StopwordModification()]
        #
        # Maximum words perturbed percentage of 20%
        #
        constraints.append(MaxWordsPerturbed(max_percent=0.2))
        #
        # Maximum word embedding euclidean distance δ of 0.5.
        #
        constraints.append(
            WordEmbeddingDistance(max_mse_dist=0.5, compare_against_original=False)
        )
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Perform word substitution with an improved genetic algorithm.
        # Fix the hyperparameter values to S = 60, M = 20, λ = 5."
        #
        search_method = ImprovedGeneticAlgorithm(
            pop_size=60,
            max_iters=20,
            max_replace_times_per_index=5,
            post_crossover_check=False,
        )

        return Attack(goal_function, constraints, transformation, search_method)
