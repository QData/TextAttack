from __future__ import annotations
from .word_swap_differential_evolution import WordSwapDifferentialEvolution
from typing import List, Tuple, Union
from textattack.shared import AttackedText
from dataclasses import dataclass

class WordSwapReorderings(WordSwapDifferentialEvolution):
    """
    Generates visually identical reorderings of a string using swap and encoding procedures.
    
    Based off of Bad Characters: Imperceptible NLP Attacks (Boucher et al., 2021).
    https://arxiv.org/abs/2106.09898 
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.PDF = chr(0x202C)
        self.LRE = chr(0x202A)
        self.RLE = chr(0x202B)
        self.LRO = chr(0x202D)
        self.RLO = chr(0x202E)
        self.PDI = chr(0x2069)
        self.LRI = chr(0x2066)
        self.RLI = chr(0x2067)

    @dataclass(eq=True, repr=True)
    class Swap:
        """Represents two characters to be swapped."""
        one: str
        two: str

    def natural(self, x: float) -> int:
        """Helper function that rounds float to the nearest natural number (positive int)"""
        return max(0, round(float(x)))

    def get_bounds(self, current_text: AttackedText, max_perturbs: int, _) -> List[Tuple[int, int]]:
        return [(-1, len(current_text.text) - 1)] * max_perturbs

    def _apply_swaps(self, elements: List[Union[str, Swap]]) -> str:
        res = ""
        for el in elements:
            if isinstance(el, self.Swap):
                res += self._apply_swaps([
                    self.LRO, self.LRI, self.RLO, self.LRI,
                    el.one, self.PDI, self.LRI, el.two,
                    self.PDI, self.PDF, self.PDI, self.PDF
                ])
            elif isinstance(el, str):  
                res += el
        return res

    def apply_perturbation(self, current_text: AttackedText, perturbation_vector: List[float], _) -> AttackedText: 
        candidate: List[Union[str, self.Swap]] = list(current_text.text)
        for perturb in map(self.natural, perturbation_vector):
            if (perturb >= 0 and len(candidate) >= 2):
                perturb = min(perturb, len(candidate) - 2)
                candidate = candidate[:perturb] + [self.Swap(candidate[perturb+1], candidate[perturb])] + candidate[perturb+2:]
        return AttackedText(self._apply_swaps(candidate))

    def _get_replacement_words(self, word: str) -> List[str]:
        candidate_words = []
        chars: List[Union[str, self.Swap]] = list(word)
        for i in range(len(chars) - 1):
            perturbed = chars[:]
            perturbed[i:i+2] = [self.Swap(chars[i+1], chars[i])]
            transformed = self._apply_swaps(perturbed)
            candidate_words.append(transformed)
        return candidate_words