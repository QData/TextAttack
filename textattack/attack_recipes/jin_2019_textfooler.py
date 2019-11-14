'''
    Jin, D., Jin, Z., Zhou, J.T., & Szolovits, P. (2019). 
    
    Is BERT Really Robust? Natural Language Attack on Text Classification and 
        Entailment. 
    
    ArXiv, abs/1907.11932.
    
'''

from textattack.attacks.blackbox import GreedyWordSwapWIR
from textattack.constraints.semantics import UniversalSentenceEncoder
from textattack.transformations import WordSwapEmbedding

def Jin2019TextFooler(model):
    #
    # Swap words with their embedding nearest-neighbors. 
    #
    # Embedding: Counter-fitted Paragram Embeddings.
    #
    # 50 nearest-neighbors with a cosine similarity of at least 0.7.
    #
    transformation = WordSwapEmbedding(max_candidates=50, min_cos_sim=0.7)
    #
    # Greedily swap words with "Word Importance Ranking".
    #
    attack = GreedyWordSwapWIR(model, transformation)
    #
    # Universal Sentence Encoder with ε = 0.9
    #
    attack.add_constraint(UniversalSentenceEncoder(0.9, metric='cosine'))
    
    return attack