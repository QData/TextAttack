"""
    Ebrahimi, J. et al. (2017)
    
    HotFlip: White-Box Adversarial Examples for Text Classification
    
    EMNLP. 
    
    ArXiv, abs/1801.00554.
    
    This is a reproduction of the HotFlip word-level attack (section 5 of the 
    paper).
"""

from textattack.goal_functions import UntargetedClassification
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.overlap import WordsPerturbed
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.search_methods import BeamSearch
from textattack.transformations import GradientBasedWordSwap

def HotFlipEbrahimi2017(model):
    #
    # "HotFlip ... uses the gradient with respect to a one-hot input 
    # representation to efficiently estimate which individual change has the 
    # highest estimated loss."
    transformation = GradientBasedWordSwap(model, top_n=1, replace_stopwords=False)
    constraints = []
    #
    # 0. "We were able to create only 41 examples (2% of the correctly-
    # classified instances of the SST test set) with one or two flips."
    # 
    constraints.append(
        WordsPerturbed(max_num_words=2)
    )
    #
    # 1. "The cosine similarity between the embedding of words is bigger than a 
    #   threshold (0.8)."
    #
    constraints.append(
            WordEmbeddingDistance(min_cos_sim=0.8)
    )
    #
    # 2. "The two words have the same part-of-speech."
    #
    constraints.append(PartOfSpeech())
    #
    # Goal is untargeted classification
    #
    goal_function = UntargetedClassification(model)
    #
    # "HotFlip ... uses a beam search to find a set of manipulations that work 
    # well together to confuse a classifier ... The adversary uses a beam size 
    # of 10."
    #
    attack = BeamSearch(goal_function, constraints=constraints,
        transformation=transformation, beam_width=10)
    
    return attack
