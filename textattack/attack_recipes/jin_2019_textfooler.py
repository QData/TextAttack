"""
    Jin, D., Jin, Z., Zhou, J.T., & Szolovits, P. (2019). 
    
    Is BERT Really Robust? Natural Language Attack on Text Classification and 
        Entailment. 
    
    ArXiv, abs/1907.11932.
    
"""

from textattack.attacks.blackbox import GreedyWordSwapWIR
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.transformations import WordSwapEmbedding

def Jin2019TextFooler(model):
    #
    # Swap words with their embedding nearest-neighbors. 
    #
    # Embedding: Counter-fitted Paragram Embeddings.
    #
    # 50 nearest-neighbors with a cosine similarity of at least 0.5.
    # (The paper claims 0.7, but analysis of the code and some empirical
    # results show that it's definitely 0.5.)
    #
    transformation = WordSwapEmbedding(max_candidates=50, check_pos=True)
    #
    # Greedily swap words with "Word Importance Ranking".
    #
    attack = GreedyWordSwapWIR(model, transformations=[transformation])
    #
    # Minimum word embedding cosine similarity of 0.5.
    #
    attack.add_constraint(
            WordEmbeddingDistance(min_cos_sim=0.5)
    )
    #
    # Universal Sentence Encoder with a minimum angular similarity of ε = 0.7.
    #
    # In the TextFooler code, they forget to divide the angle between the two
    # embeddings by pi. So if the original threshold was that 1 - sim >= 0.7, the 
    # new threshold is 1 - (0.3) / pi = 0.90445
    #
    use_constraint = UniversalSentenceEncoder(threshold=0.904458599, 
        metric='angular', compare_with_original=False, window_size=15,
        skip_text_shorter_than_window=True)
    attack.add_constraint(use_constraint)
    
    return attack
