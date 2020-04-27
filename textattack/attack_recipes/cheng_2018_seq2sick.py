"""
    Cheng, Minhao, et al. 
    
    Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with 
        Adversarial Examples
    
    ArXiv, abs/1803.01128.
    
"""

from textattack.attack_methods import GreedyWordSwapWIR
from textattack.constraints.overlap import LevenshteinEditDistance
from textattack.goal_functions import NonOverlappingOutput
from textattack.transformations import WordSwapRandomCharacterSubstitution

def Cheng2018Seq2Sick(model, goal_function='non_overlapping'):
    #
    # Goal is non-overlapping output.
    # @TODO verify goal function fits model.
    #
    goal_function = NonOverlappingOutput(model)
    # @TODO implement transformation / search method just like they do in
    # seq2sick.
    transformation = WordSwapRandomCharacterSubstitution()
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
