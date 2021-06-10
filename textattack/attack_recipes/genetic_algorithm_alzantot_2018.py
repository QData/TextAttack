"""

Alzantot Genetic Algorithm
=======================================
(Generating Natural Language Adversarial Examples)

.. warning::
    This attack uses a very slow language model. Consider using the ``fast-alzantot``
    recipe instead.

"""

from textattack import Attack
from textattack.constraints.grammaticality.language_models import (
    Google1BillionWordsLanguageModel,
)
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import AlzantotGeneticAlgorithm
from textattack.transformations import WordSwapEmbedding

from .attack_recipe import AttackRecipe


class GeneticAlgorithmAlzantot2018(AttackRecipe):
    """Alzantot, M., Sharma, Y., Elgohary, A., Ho, B., Srivastava, M.B., &
    Chang, K. (2018).

    Generating Natural Language Adversarial Examples.

    https://arxiv.org/abs/1804.07998
    """

    @staticmethod
    def build(model_wrapper):
        #
        # Swap words with their embedding nearest-neighbors.
        #
        # Embedding: Counter-fitted Paragram Embeddings.
        #
        # "[We] fix the hyperparameter values to S = 60, N = 8, K = 4, and δ = 0.5"
        #
        transformation = WordSwapEmbedding(max_candidates=8)
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        #
        # Maximum words perturbed percentage of 20%
        #
        constraints.append(MaxWordsPerturbed(max_percent=0.2))
        #
        # Maximum word embedding euclidean distance of 0.5.
        #
        constraints.append(
            WordEmbeddingDistance(max_mse_dist=0.5, compare_against_original=False)
        )
        #
        # Language Model
        #
        constraints.append(
            Google1BillionWordsLanguageModel(
                top_n_per_index=4, compare_against_original=False
            )
        )
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Perform word substitution with a genetic algorithm.
        #
        search_method = AlzantotGeneticAlgorithm(
            pop_size=60, max_iters=20, post_crossover_check=False
        )

        return Attack(goal_function, constraints, transformation, search_method)
