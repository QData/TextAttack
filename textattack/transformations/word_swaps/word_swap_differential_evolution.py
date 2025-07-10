"""
Word Swap for Differential Evolution
-------------------------------------
Extends WordSwap. 

If a Transformation wants to be compatible with 
textattack.search_methods.DifferentialEvolution,
then it must extend from this class.
"""
from textattack.transformations.word_swaps import WordSwap
from textattack.shared import AttackedText
from typing import List, Tuple, Optional, Any

class WordSwapDifferentialEvolution(WordSwap):
    """
    A base class for Word Swaps compatible with Differential Evolution search.

    Subclasses must implement `_get_bounds` and `apply_perturbation`. Implementing `_get_precomputed` is optional.
    """

    def _get_bounds(self, current_text: AttackedText, max_perturbs: int, precomputed: Optional[List[Any]]) -> List[Tuple[int, int]]:
        raise NotImplementedError()

    def get_bounds_and_precomputed(self, current_text: AttackedText, max_perturbs: int) -> Tuple[List[Tuple[int, int]], Optional[List[Any]]]:
        """
        Returns the bounds and optional precomputed values for differential evolution.

        If the subclass implements `_get_precomputed(current_text)`, it will be used to generate
        precomputed values; otherwise, `precomputed` will be `None`.

        Args:
            current_text (AttackedText): The text being attacked.
            max_perturbs (int): Maximum number of word swaps allowed.

        Returns:
            Tuple[List[Tuple[int, int]], Optional[List[Any]]]: Bounds and optional precomputed data.
        """
        if hasattr(self, "_get_precomputed"):
            precomputed = self._get_precomputed(current_text)
        else:
            precomputed = None
        bounds = self._get_bounds(current_text, max_perturbs, precomputed)
        return (bounds, precomputed)

    def apply_perturbation(self, current_text: AttackedText, perturbation_vector: List[float], precomputed: Optional[List[Any]]):
        """
        Applies a perturbation to the input text based on a perturbation vector.

        Args:
            current_text (AttackedText): The original text to perturb.
            perturbation_vector (List[float]): The vector encoding the perturbation.
            precomputed (Optional[List[Any]]): Optional precomputed data to speed up perturbation.

        Returns:
            AttackedText: The perturbed version of the input text.

        Note:
            Subclasses must implement this method to define how the perturbation vector is applied.
        """
        raise NotImplementedError()

    