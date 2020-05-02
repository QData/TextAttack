from .beam_search import BeamSearch

class GreedyWordSwap(BeamSearch):
    """ 
    An attack that greedily chooses from a list of possible perturbations.
    
    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation: The type of transformation.
        max_words_changed (:obj:`int`, optional): The maximum number of words 
            to change.
        
    """
    def __init__(self, goal_function, transformation, constraints=[], max_words_changed=32):
        super().__init__(goal_function, transformation, constraints=constraints, 
            beam_width=1, max_words_changed=max_words_changed)
