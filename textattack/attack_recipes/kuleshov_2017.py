"""
    Kuleshov, V. et al. 
    
    Generating Natural Language Adversarial Examples. 
    
    
    https://openreview.net/pdf?id=r1QZ3zbAZ.
"""

from textattack.constraints.overlap import WordsPerturbed
from textattack.constraints.grammaticality.language_models import GPT2
from textattack.constraints.semantics.sentence_encoders import ThoughtVector
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwap
from textattack.transformations import WordSwapEmbedding

def Kuleshov2017(model):
    #
    # "Specifically, in all experiments, we used a target of τ = 0.7,
    # a neighborhood size of N = 15, and parameters λ_1 = 0.2 and δ = 0.5; we set
    # the syntactic bound to λ_2 = 2 nats for sentiment analysis"
    
    #
    # Word swap with top-15 counter-fitted embedding neighbors.
    #
    transformation = WordSwapEmbedding(max_candidates=15)
    #
    # Maximum of 50% of words perturbed (δ in the paper).
    #
    constraints = []
    constraints.append(
            WordsPerturbed(max_percent=0.5)
    )
    #
    # Maximum thought vector Euclidean distance of λ_1 = 0.2. (eq. 4)
    #
    constraints.append(
        ThoughtVector(embedding_type='paragramcf', threshold=0.2, metric='max_euclidean')
    )
    #
    #
    # Maximum language model log-probability difference of λ_2 = 2. (eq. 5)
    #
    constraints.append(
        GPT2(max_log_prob_diff=2.0)
    )
    #
    # Goal is untargeted classification: reduce original probability score 
    # to below τ = 0.7 (Algorithm 1).
    #
    goal_function = UntargetedClassification(model, target_max_score=0.7)
    #
    # Perform word substitution with a genetic algorithm.
    #
    attack = GreedyWordSwap(goal_function, constraints=constraints,
        transformation=transformation)
    
    return attack


        # GPT2(max_log_prob_diff=2)