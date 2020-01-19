"""
    Alzantot, M., Sharma, Y., Elgohary, A., Ho, B., Srivastava, M.B., & Chang, 
        K. (2018). 
    
    Generating Natural Language Adversarial Examples. 
    
    EMNLP. 
    
    ArXiv, abs/1801.00554.
"""

from textattack.attack_methods import GeneticAlgorithm
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.language_models import GoogleLanguageModel
from textattack.transformations import WordSwapEmbedding

def Alzantot2018GeneticAlgorithm(model):
    #
    # Swap words with their embedding nearest-neighbors. 
    #
    # Embedding: Counter-fitted Paragram Embeddings.
    #
    # "[We] fix the hyperparameter values to S = 60, N = 8, K = 4, and δ = 0.5"
    #
    transformation = WordSwapEmbedding(max_candidates=8)
    #
    # Perform word substitution with a genetic algorithm.
    #
    attack = GeneticAlgorithm(model, transformation=transformation, 
        pop_size=60, max_iters=20)
    #
    # Maximum word embedding euclidean distance of 0.5.
    #
    attack.add_constraint(
            WordEmbeddingDistance(max_mse_dist=0.5)
    )
    #
    # Language Model
    #
    attack.add_constraint(
            GoogleLanguageModel(top_n_per_index=4)
    )
    
    return attack
