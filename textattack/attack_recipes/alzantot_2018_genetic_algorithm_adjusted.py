"""
    Alzantot, M., Sharma, Y., Elgohary, A., Ho, B., Srivastava, M.B., & Chang, 
        K. (2018). 
    
    Generating Natural Language Adversarial Examples. 
    
    EMNLP. 
    
    ArXiv, abs/1801.00554.
"""

from textattack.attacks.blackbox import GeneticAlgorithm
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder, BERT
from textattack.constraints.syntax import PartOfSpeech, LanguageTool
from textattack.transformations import WordSwapEmbedding

def Alzantot2018GeneticAlgorithmAdjusted(model, SE_thresh=0.98, sentence_encoder='bert'):
    #
    # Swap words with their embedding nearest-neighbors. 
    #
    # Embedding: Counter-fitted PARAGRAM-SL999 vectors.
    #
    # "[We] fix the hyperparameter values to S = 60, N = 8, K = 4, and δ = 0.5"
    #
    transformation = WordSwapEmbedding(max_candidates=50, textfooler_stopwords=True)
    #
    # Greedily swap words with "Word Importance Ranking".
    #
    attack = GeneticAlgorithm(model, transformations=[transformation], 
        pop_size=60, max_iters=20)
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
