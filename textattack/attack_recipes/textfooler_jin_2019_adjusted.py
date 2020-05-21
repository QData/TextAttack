from textattack.shared.attack import Attack
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder, BERT
from textattack.constraints.grammaticality import PartOfSpeech, LanguageTool
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapEmbedding

def TextFoolerJin2019Adjusted(model, SE_thresh=0.98, sentence_encoder='bert'):
    """
        Jin, D., Jin, Z., Zhou, J.T., & Szolovits, P. (2019). 
        
        Is BERT Really Robust? Natural Language Attack on Text Classification and Entailment. 
        
        https://arxiv.org/abs/1907.11932 
       
        Constraints adjusted from paper to align with human evaluation.
    """
    #
    # Swap words with their embedding nearest-neighbors. 
    #
    # Embedding: Counter-fitted PARAGRAM-SL999 vectors.
    #
    # 50 nearest-neighbors with a cosine similarity of at least 0.5.
    # (The paper claims 0.7, but analysis of the code and some empirical
    # results show that it's definitely 0.5.)
    #
    transformation = WordSwapEmbedding(max_candidates=50)
    #
    # Don't modify the same word twice or stopwords
    #
    constraints = [
        RepeatModification(),
        StopwordModification()
    ]
    #
    # Minimum word embedding cosine similarity of 0.9.
    #
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
    # Untargeted attack   
    #
    goal_function = UntargetedClassification(model)

    #
    # Greedily swap words with "Word Importance Ranking".
    #
    search_method = GreedyWordSwapWIR()
    
    return Attack(goal_function, constraints, transformation, search_method)
