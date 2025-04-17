"""
Word Swap by Invisible Characters
-------------------------------
"""

from .word_swap import WordSwap
from typing import List, Tuple
from textattack.shared import AttackedText
import random

class WordSwapInvisibleCharacters(WordSwap):
    """
    Transforms an input by replacing its words with visually similar words
    by injecting invisible characters.

    Based off of Bad Characters: Imperceptible NLP Attacks (Boucher et al., 2021).
    https://arxiv.org/abs/2106.09898 
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.invisible_chars = ["\u200B", "\u200C", "\u200D"]

    def bounds(self, sentence: AttackedText, max_perturbs: int) -> List[Tuple[int, int]]:
        return [(0, len(self.invisible_chars) - 1), (-1, len(sentence.text) - 1)] * max_perturbs

    def _get_replacement_words(self, word: str) -> List[str]:
        candidate_words = []
        for i in range(1, len(word)):  
            for inv_char in self.invisible_chars:
                new_word = word[:i] + inv_char + word[i:]
                candidate_words.append(new_word)
        return candidate_words

    def natural(self, x: float) -> int:
        return max(0, round(float(x)))

    def apply_perturbation(self, sentence: AttackedText, perturbation_vector: List[float]) -> AttackedText:
        candidate = list(sentence.text)
        for i in range(0, len(perturbation_vector), 2):
            inp_index = self.natural(perturbation_vector[i+1])
            if (inp_index >= 0):
                inv_char = self.invisible_chars[self.natural(perturbation_vector[i])]
                candidate = candidate[:inp_index] + [inv_char] + candidate[inp_index:]
        return AttackedText(''.join(candidate))
