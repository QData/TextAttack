from textattack.constraints.overlap import LevenshteinEditDistance
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.goal_functions import NonOverlappingOutput
from textattack.search_methods import GreedyWordSwapWIR
from textattack.shared.attack import Attack
from textattack.transformations import WordSwapEmbedding


def Seq2SickCheng2018BlackBox(model, goal_function="non_overlapping"):
    """
        Cheng, Minhao, et al. 
        
        Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with 
        Adversarial Examples
        
        https://arxiv.org/abs/1803.01128    
        
        This is a greedy re-implementation of the seq2sick attack method. It does 
        not use gradient descent.
    """

    #
    # Goal is non-overlapping output.
    #
    goal_function = NonOverlappingOutput(model)
    # @TODO implement transformation / search method just like they do in
    # seq2sick.
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
    search_method = GreedyWordSwapWIR()

    return Attack(goal_function, constraints, transformation, search_method)
