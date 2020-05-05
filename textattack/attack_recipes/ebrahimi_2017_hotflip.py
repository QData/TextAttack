"""
    Ebrahimi, J. et al. (2017)
    
    HotFlip: White-Box Adversarial Examples for Text Classification
    
    EMNLP. 
    
    ArXiv, abs/1801.00554.
"""

from textattack.attack_methods import GeneticAlgorithm
from textattack.constraints.overlap import WordsPerturbedPercentage
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
    # 1. The cosine similarity between the embedding of words is bigger than a 
    #   threshold (0.8).
    #
    constraints.append(
            WordsPerturbedPercentage(max_percent=20)
    )
    #
    # 2. The two words have the same part-of-speech.
    #
    constraints.append(PartOfSpeech())
    #
    # Language Model
    #
    constraints.append(
            GoogleLanguageModel(top_n_per_index=4)
    )
    #
    # Goal is untargeted classification
    #
    goal_function = UntargetedClassification(model)
    #
    # Perform word substitution with a genetic algorithm.
    #
    attack = GeneticAlgorithm(goal_function, constraints=constraints,
        transformation=transformation, pop_size=60, max_iters=20)
    
    return attack
