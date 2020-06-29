from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import PSOAlgorithm
from textattack.shared.attack import Attack
from textattack.transformations import WordSwapEmbedding, WordSwapHowNet


def PSOZang2020(model):
    """
        Zang, Y., Yang, C., Qi, F., Liu, Z., Zhang, M., Liu, Q., & Sun, M. (2019).
        
        Word-level Textual Adversarial Attacking as Combinatorial Optimization.
        
        https://www.aclweb.org/anthology/2020.acl-main.540.pdf
    """
    #
    # Swap words with their embedding nearest-neighbors.
    #
    # Embedding: Counter-fitted Paragram Embeddings.
    #
    # "[We] fix the hyperparameter values to S = 60, N = 8, K = 4, and δ = 0.5"
    #
    # transformation = WordSwapEmbedding(max_candidates=20)
    transformation = WordSwapHowNet()
    #

    # Don't modify the same word twice or stopwords
    #
    constraints = [RepeatModification(), StopwordModification()]
    #
    #
    # constraints.append(MaxWordsPerturbed(max_percent=0.20))
    #
    # Maximum word embedding euclidean distance of 0.5.
    #
    # constraints.append(WordEmbeddingDistance(max_mse_dist=0.5))
    # constraints.append(PartOfSpeech(tagger_type='flair', allow_verb_noun_swap=True))
    #
    # Goal is untargeted classification
    #
    goal_function = UntargetedClassification(model)
    #
    # Perform word substitution with a genetic algorithm.
    #
    search_method = PSOAlgorithm(pop_size=60, max_iters=20)

    return Attack(goal_function, constraints, transformation, search_method)
