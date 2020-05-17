from .beam_search import BeamSearch

class GreedySearch(BeamSearch):
    """ 
    An attack that greedily chooses from a list of possible perturbations.

    """
    def __init__(self):
        super().__init__(beam_width=1)

    def extra_repr_keys(self):
        return []
