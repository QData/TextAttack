"""
Word Swap for Differential Evolution
-------------------------------
Extends WordSwap. If a Transformation wants to be compatible with 
    textattack.search_methods.DifferentialEvolution
then it must extend from this class.

"""
from textattack.transformations.word_swaps import WordSwap
from textattack.shared import AttackedText
from typing import List, Tuple, Optional, Any

class WordSwapDifferentialEvolution(WordSwap):
    
    def get_precomputed(self, current_text: AttackedText) -> Optional[List[Any]]:
        return None
        
    def get_bounds(self, current_text: AttackedText, max_perturbs: int, precomputed: Optional[List[Any]]) -> List[Tuple[int, int]]:
        raise NotImplementedError()

    def get_bounds_and_precomputed(self, current_text: AttackedText, max_perturbs: int) -> Tuple[List[Tuple[int, int]], Optional[List[Any]]]:
        precomputed = self.get_precomputed(current_text)
        bounds = self.get_bounds(current_text, max_perturbs, precomputed)
        return (bounds, precomputed)

    def apply_perturbation(self, current_text: AttackedText, perturbation_vector: List[float], precomputed: Optional[List[Any]]):
        raise NotImplementedError()

    