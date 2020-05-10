"""
    Cheng, Minhao, et al. 
    
    Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with 
        Adversarial Examples
    
    ArXiv, abs/1803.01128.
    
    
    This is a greedy re-implementation of the seq2sick attack method. It does 
        not use gradient descent.
    
"""

from textattack.constraints.overlap import LevenshteinEditDistance
from textattack.goal_functions import NonOverlappingOutput
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapEmbedding

def Seq2SickCheng2018BlackBox(model, goal_function='non_overlapping'):
    #
    # Goal is non-overlapping output.
    #
    goal_function = NonOverlappingOutput(model)
    # @TODO implement transformation / search method just like they do in
    # seq2sick.
    transformation = WordSwapEmbedding(max_candidates=50)
    #
    # In these experiments, we hold the maximum difference
    # on edit distance (ϵ) to a constant 30 for each sample.
    #
    #
    # Greedily swap words with "Word Importance Ranking".
    #
    attack = GreedyWordSwapWIR(goal_function, transformation=transformation,
        constraints=[], max_depth=10)
    
    return attack
