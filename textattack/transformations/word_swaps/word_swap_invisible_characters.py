"""
Word Swap by Invisible Characters
-----------------------------------
"""

from .word_swap_differential_evolution import WordSwapDifferentialEvolution
from typing import List, Tuple
from textattack.shared import AttackedText
import random
import numpy as np

class WordSwapInvisibleCharacters(WordSwapDifferentialEvolution):
    """
    Transforms an input by replacing its words with visually similar words
    by injecting invisible characters.

    Based off of Bad Characters: Imperceptible NLP Attacks (Boucher et al., 2021).
    https://arxiv.org/abs/2106.09898 
    """

    def __init__(self, random_one=False, **kwargs):
        super().__init__(**kwargs)
        self.invisible_chars = ["\u200B", "\u200C", "\u200D"]
        self.random_one = random_one

    def _get_bounds(self, current_text: AttackedText, max_perturbs: int, _) -> List[Tuple[int, int]]:
        return [(0, len(self.invisible_chars) - 1), (-1, len(current_text.text) - 1)] * max_perturbs

    def _natural(self, x: float) -> int:
        """Helper function that rounds float to the nearest natural number (positive int)"""
        return max(0, round(float(x)))

    def apply_perturbation(self, current_text: AttackedText, perturbation_vector: List[float], _) -> AttackedText:
        candidate = list(current_text.text)
        for i in range(0, len(perturbation_vector), 2):
            inp_index = self._natural(perturbation_vector[i+1])
            if (inp_index >= 0):
                inv_char = self.invisible_chars[self._natural(perturbation_vector[i])]
                candidate = candidate[:inp_index] + [inv_char] + candidate[inp_index:]
        return AttackedText(''.join(candidate))

    def _get_replacement_words(self, word: str) -> List[str]:
        candidate_words = []
        if self.random_one:
            if len(word) <= 1:
                return []
            i = np.random.randint(1, len(word))  # insert between characters
            inv_char = np.random.choice(self.invisible_chars)
            new_word = word[:i] + inv_char + word[i:]
            candidate_words.append(new_word)
        else:
            for i in range(1, len(word)):  # start at 1 to avoid invisible prefix
                for inv_char in self.invisible_chars:
                    new_word = word[:i] + inv_char + word[i:]
                    candidate_words.append(new_word)
        return candidate_words

    @property
    def deterministic(self):
        return not self.random_one
    
    def extra_repr_keys(self):
        return super().extra_repr_keys()