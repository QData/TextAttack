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
from textattack.search_methods import GeneticAlgorithm
from textattack.shared.attack import Attack
from textattack.transformations import WordSwapEmbedding


def Alzantot2018(model):
    """
        Alzantot, M., Sharma, Y., Elgohary, A., Ho, B., Srivastava, M.B., & Chang, K. (2018). 
        
        Generating Natural Language Adversarial Examples. 
        
        https://arxiv.org/abs/1801.00554 
    """
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
    constraints.append(Google1BillionWordsLanguageModel(top_n_per_index=4))
    #
    # Goal is untargeted classification
    #
    goal_function = UntargetedClassification(model)
    #
    # Perform word substitution with a genetic algorithm.
    #
    search_method = GeneticAlgorithm(pop_size=60, max_iters=20)

    return Attack(goal_function, constraints, transformation, search_method)
