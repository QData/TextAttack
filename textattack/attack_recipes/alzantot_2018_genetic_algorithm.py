'''
    Alzantot, M., Sharma, Y., Elgohary, A., Ho, B., Srivastava, M.B., & Chang, 
        K. (2018). 
    
    Generating Natural Language Adversarial Examples. 
    
    EMNLP. 
    
    ArXiv, abs/1801.00554.
'''

from textattack.attacks.blackbox import GeneticAlgorithm
from textattack.constraints.semantics import GoogleLanguageModel
from textattack.transformations import WordSwapEmbedding

def Alzantot2018GeneticAlgorithm(model):
    #
    # Swap words with their embedding nearest-neighbors. 
    #
    # Embedding: Counter-fitted Paragram Embeddings.
    #
    # 50 nearest-neighbors with a cosine similarity of at least 0.7.
    #
    transformation = WordSwapEmbedding()
    #
    # Perform word substitution with a genetic algorithm.
    #
    attack = GeneticAlgorithm(model, transformation)
    #
    # Language Model (TODO-- what threshold?)
    #
    attack.add_constraint(
            GoogleLanguageModel()
    )
    
    return attack