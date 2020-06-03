from textattack.search_methods import GreedyWordSwapWIR
from textattack.shared.attack import Attack
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.transformations import CompositeTransformation, WordDeletion
from textattack.goal_functions import UntargetedClassification

def InputReductionFeng2018(model):
    """
    Feng, Wallace, Grissom, Iyyer, Rodriguez, Boyd-Graber. (2018).
    
    Pathologies of Neural Models Make Interpretations Difficult.
 
    ArXiv, abs/1804.07781.
    """
    transformation = WordDeletion()
    
    constraints = [
        RepeatModification(),
        StopwordModification()
    ]
    #
    # Goal is untargeted classification
    #
    goal_function = UntargetedClassification(model)
    #
    # Greedily swap words with "Word Importance Ranking".
    #
    search_method = GreedyWordSwapWIR()
        
    return Attack(goal_function, constraints, transformation, search_method)
