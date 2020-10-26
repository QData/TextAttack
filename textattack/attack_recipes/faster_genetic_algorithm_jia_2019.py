"""

Faster Alzantot Genetic Algorithm
===================================
(Certified Robustness to Adversarial Word Substitutions)


"""

from textattack.constraints.grammaticality.language_models import (
    LearningToWriteLanguageModel,
)
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import AlzantotGeneticAlgorithm
from textattack.shared.attack import Attack
from textattack.transformations import WordSwapEmbedding

from .attack_recipe import AttackRecipe


class FasterGeneticAlgorithmJia2019(AttackRecipe):
    """Certified Robustness to Adversarial Word Substitutions.

    Robin Jia, Aditi Raghunathan, Kerem Göksel, Percy Liang (2019).

    https://arxiv.org/pdf/1909.00986.pdf
    """

    @staticmethod
    def build(model):
        #
        # Section 5: Experiments
        #
        # We base our sets of allowed word substitutions S(x, i) on the
        # substitutions allowed by Alzantot et al. (2018). They demonstrated that
        # their substitutions lead to adversarial examples that are qualitatively
        # similar to the original input and retain the original label, as judged
        # by humans. Alzantot et al. (2018) define the neighbors N(w) of a word w
        # as the n = 8 nearest neighbors of w in a “counter-fitted” word vector
        # space where antonyms are far apart (Mrksiˇ c´ et al., 2016). The
        # neighbors must also lie within some Euclidean distance threshold. They
        # also use a language model constraint to avoid nonsensical perturbations:
        # they allow substituting xi with x˜i ∈ N(xi) if and only if it does not
        # decrease the log-likelihood of the text under a pre-trained language
        # model by more than some threshold.
        #
        # We make three modifications to this approach:
        #
        # First, in Alzantot et al. (2018), the adversary
        # applies substitutions one at a time, and the
        # neighborhoods and language model scores are computed.
        # Equation (4) must be applied before the model
        # can combine information from multiple words, but it can
        # be delayed until after processing each word independently.
        # Note that the model itself classifies using a different
        # set of pre-trained word vectors; the counter-fitted vectors
        # are only used to define the set of allowed substitution words.
        # relative to the current altered version of the input.
        # This results in a hard-to-define attack surface, as
        # changing one word can allow or disallow changes
        # to other words. It also requires recomputing
        # language model scores at each iteration of the genetic
        # attack, which is inefficient. Moreover, the same
        # word can be substituted multiple times, leading
        # to semantic drift. We define allowed substitutions
        # relative to the original sentence x, and disallow
        # repeated substitutions.
        #
        # Second, we use a faster language model that allows us to query
        # longer contexts; Alzantot et al. (2018) use a slower language
        # model and could only query it with short contexts.

        # Finally, we use the language model constraint only
        # at test time; the model is trained against all perturbations in N(w). This encourages the model to be
        # robust to a larger space of perturbations, instead of
        # specializing for the particular choice of language
        # model. See Appendix A.3 for further details. [This is a model-specific
        # adjustment, so does not affect the attack recipe.]
        #
        # Appendix A.3:
        #
        # In Alzantot et al. (2018), the adversary applies replacements one at a
        # time, and the neighborhoods and language model scores are computed
        # relative to the current altered version of the input. This results in a
        # hard-to-define attack surface, as the same word can be replaced many
        # times, leading to semantic drift. We instead pre-compute the allowed
        # substitutions S(x, i) at index i based on the original x. We define
        # S(x, i) as the set of x_i ∈ N(x_i) such that where probabilities are
        # assigned by a pre-trained language model, and the window radius W and
        # threshold δ are hyperparameters. We use W = 6 and δ = 5.
        #
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
        # Maximum words perturbed percentage of 20%
        #
        constraints.append(MaxWordsPerturbed(max_percent=0.2))
        #
        # Maximum word embedding euclidean distance of 0.5.
        #
        constraints.append(WordEmbeddingDistance(max_mse_dist=0.5))
        #
        # Language Model
        #
        #
        #
        constraints.append(
            LearningToWriteLanguageModel(
                window_size=6, max_log_prob_diff=5.0, compare_against_original=True
            )
        )
        # constraints.append(LearningToWriteLanguageModel(window_size=5))
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model)
        #
        # Perform word substitution with a genetic algorithm.
        #
        search_method = AlzantotGeneticAlgorithm(
            pop_size=60, max_iters=20, post_crossover_check=False
        )

        return Attack(goal_function, constraints, transformation, search_method)
