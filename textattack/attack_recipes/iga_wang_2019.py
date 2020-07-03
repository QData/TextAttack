from textattack.constraints.grammaticality.language_models import (
    Google1BillionWordsLanguageModel,
)
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import ImprovedGeneticAlgorithm
from textattack.shared.attack import Attack
from textattack.transformations import WordSwapEmbedding


def IGAWang2019(model):
    """
        Xiaosen Wang, Hao Jin, Kun He (2019). 
        
        Natural Language Adversarial Attack and Defense inWord Level. 
        
        http://arxiv.org/abs/1909.06723 
    """
    #
    # Swap words with their embedding nearest-neighbors.
    #
    # Embedding: Counter-fitted Paragram Embeddings.
    #
    # "[We] fix the hyperparameter values to S = 60, δ = 0.5, λ = 5, and N is unrestricted (50)."
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
    # Maximum word embedding euclidean distance of 0.5.
    #
    constraints.append(WordEmbeddingDistance(max_mse_dist=0.5))
    #
    # Goal is untargeted classification
    #
    goal_function = UntargetedClassification(model)
    #
    # Perform word substitution with an improved genetic algorithm.
    #
    search_method = ImprovedGeneticAlgorithm(
        max_pop_size=60, max_iters=20, max_replaced_times=5
    )

    return Attack(goal_function, constraints, transformation, search_method)
