"""
    Ebrahimi, J. et al. (2017)
    
    HotFlip: White-Box Adversarial Examples for Text Classification
    
    EMNLP. 
    
    ArXiv, abs/1801.00554.
"""

from textattack.attack_methods import GeneticAlgorithm
from textattack.constraints.overlap import NumberOfWordsSwapped, WordsPerturbedPercentage
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.syntax import PartOfSpeech
from textattack.transformations.white_box import GradientBasedWordSwap
from textattack.goal_functions import UntargetedClassification

def Ebrahimi2017HotFlip(model):
    #
    # Swap words with their embedding nearest-neighbors. 
    #
    # Embedding: Counter-fitted Paragram Embeddings.
    #
    # "[We] fix the hyperparameter values to S = 60, N = 8, K = 4, and δ = 0.5"
    #
    transformation = GradientBasedWordSwap(model,  ???top_n=1000?, replace_stopwords=False)
    constraints = []
    #
    # [0.] We were able to create only 41 examples (2% of the correctly-
    # classified instances of the SST test set) with one or two flips.
    # 
    constraints.append(
        NumberOfWordsSwapped(max_swaps=2)
    )
    #
    # 1. The cosine similarity between the embedding of words is bigger than a 
    #   threshold (0.8).
    #
    constraints.append(
            WordEmbeddingDistance(min_cos_sim=0.8)
    )
    #
    # 2. The two words have the same part-of-speech.
    #
    constraints.append(PartOfSpeech())
    #
    # Goal is untargeted classification
    #
    goal_function = UntargetedClassification(model)
    #
    # HotFlip ... uses a beam search to find a set of manipulations that work 
    # well together to confuse a classifier ... The adversary uses a beam size 
    # of 10.
    #
    attack = BeamSearch(goal_function, constraints=constraints,
        transformation=transformation, beam_width=10)
    
    return attack
