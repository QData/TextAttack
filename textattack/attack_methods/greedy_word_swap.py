from .attack import Attack
from .beam_search import BeamSearch
from textattack.attack_results import AttackResult, FailedAttackResult

class GreedyWordSwap(BeamSearch):
    """ 
    An attack that greedily chooses from a list of possible perturbations.
    
    Args:
        model: The model to attack.
        transformation: The type of transformation.
        max_words_changed (:obj:`int`, optional): The maximum number of words 
            to change.
        
    """
    def __init__(self, model, transformation, constraints=[], max_words_changed=32):
        super().__init__(model, transformation, constraints=constraints, 
            beam_width=1, max_words_changed=32)