"""
    Jin, D., Jin, Z., Zhou, J.T., & Szolovits, P. (2019). 
    
    Is BERT Really Robust? Natural Language Attack on Text Classification and 
        Entailment. 
    
    ArXiv, abs/1907.11932.
    
"""

from textattack.attack_methods import GreedyWordSwapWIR
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder, BERT
from textattack.constraints.syntax import PartOfSpeech, LanguageTool
from textattack.transformations import WordSwapEmbedding

def Jin2019TextFoolerAdjusted(model, SE_thresh=0.98, sentence_encoder='use'):
    #
    # Swap words with their embedding nearest-neighbors. 
    #
    # Embedding: Counter-fitted PARAGRAM-SL999 vectors.
    #
    # 50 nearest-neighbors with a cosine similarity of at least 0.5.
    # (The paper claims 0.7, but analysis of the code and some empirical
    # results show that it's definitely 0.5.)
    #
    transformation = WordSwapEmbedding(max_candidates=50, textfooler_stopwords=True)
    #
    # Greedily swap words with "Word Importance Ranking".
    #
    attack = GreedyWordSwapWIR(model, transformations=transformation,
        max_depth=None)
    #
    # Minimum word embedding cosine similarity of 0.9.
    #
    attack.add_constraint(
            WordEmbeddingDistance(min_cos_sim=0.9)
    )
    #
    # Universal Sentence Encoder with a minimum angular similarity of ε = 0.7.
    #
    # In the TextFooler code, they forget to divide the angle between the two
    # embeddings by pi. So if the original threshold was that 1 - sim >= 0.7, the 
    # new threshold is 1 - (0.3) / pi = 0.90445
    #
    if sentence_encoder == 'bert':
        se_constraint = BERT(threshold=SE_thresh,
            metric='cosine', compare_with_original=False, window_size=15,
            skip_text_shorter_than_window=False)
    else:
        se_constraint = UniversalSentenceEncoder(threshold=SE_thresh,
            metric='cosine', compare_with_original=False, window_size=15,
            skip_text_shorter_than_window=False)
    attack.add_constraint(se_constraint)
    #
    # Do grammar checking
    #
    attack.add_constraint(
            LanguageTool(0)
    )
    
    return attack
