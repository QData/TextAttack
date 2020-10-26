"""
Greedy Search
=================
"""
from .beam_search import BeamSearch


class GreedySearch(BeamSearch):
    """A search method that greedily chooses from a list of possible
    perturbations.

    Implemented by calling ``BeamSearch`` with beam_width set to 1.
    """

    def __init__(self):
        super().__init__(beam_width=1)

    def extra_repr_keys(self):
        return []
