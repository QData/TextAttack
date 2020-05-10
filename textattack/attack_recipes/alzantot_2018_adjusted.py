"""
    Alzantot, M., Sharma, Y., Elgohary, A., Ho, B., Srivastava, M.B., & Chang, 
        K. (2018). 
    
    Generating Natural Language Adversarial Examples. 
    
    EMNLP. 
    
    ArXiv, abs/1801.00554.
"""

from textattack.constraints.grammaticality import PartOfSpeech, LanguageTool
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder, BERT
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GeneticAlgorithm
from textattack.transformations import WordSwapEmbedding

def Alzantot2018Adjusted(model, SE_thresh=0.98, sentence_encoder='bert'):
    #
    # Swap words with their embedding nearest-neighbors. 
    #
    # Embedding: Counter-fitted PARAGRAM-SL999 vectors.
    #
    # "[We] fix the hyperparameter values to S = 60, N = 8, K = 4, and δ = 0.5"
    #
    transformation = WordSwapEmbedding(max_candidates=50, textfooler_stopwords=True)
    #
    # Minimum word embedding cosine similarity of 0.9.
    #
    constraints = []
    constraints.append(
            WordEmbeddingDistance(min_cos_sim=0.9)
    )
    #
    # Universal Sentence Encoder with a minimum angular similarity of ε = 0.7.
    #
    if sentence_encoder == 'bert':
        se_constraint = BERT(threshold=SE_thresh,
            metric='cosine', compare_with_original=False, window_size=15,
            skip_text_shorter_than_window=False)
    else:
        se_constraint = UniversalSentenceEncoder(threshold=SE_thresh,
            metric='cosine', compare_with_original=False, window_size=15,
            skip_text_shorter_than_window=False)
    constraints.append(se_constraint)
    #
    # Do grammar checking
    #
    constraints.append(
            LanguageTool(0)
    )
    #
    # Goal is untargeted classification
    #
    goal_function = UntargetedClassification(model)
    #
    # Greedily swap words with "Word Importance Ranking".
    #
    attack = GeneticAlgorithm(goal_function, transformation=transformation, 
        constraints=constraints, pop_size=60, max_iters=20)
    return attack
