import random
from .word_swap import WordSwap

from typing import List
from textattack.shared import AttackedText

class WordSwapDeletions(WordSwap):
    """
    Generates text transformations by embedding Unicode control characters 
    (e.g., backspace, delete, carriage return) into strings to manipulate
    how they render or interact with models.
    """

    def __init__(self):
        super().__init__()
        self.del_chr = chr(0x8)
        self.ins_chr_min = '!'
        self.ins_chr_max = '~'

    def bounds(self, sentence, max_perturbs):
        return [(-1, len(sentence.text) - 1), (ord(self.ins_chr_min), ord(self.ins_chr_max))] * max_perturbs

    def _get_replacement_words(self, word):
        candidate_words = []
        return candidate_words

    def natural(self, x: float) -> int:
        """Rounds float to the nearest natural number (positive int)"""
        return max(0, round(float(x)))
    
    def apply_perturbation(self, sentence, perturbation_vector: List[float]): # AttackedText object to AttackedText object
        candidate = list(sentence.text)
        for i in range(0, len(perturbation_vector), 2):
            idx = self.natural(perturbation_vector[i])
            char = chr(self.natural(perturbation_vector[i+1]))
            candidate = candidate[:idx] + [char, self.del_chr] + candidate[idx:]
            for j in range(i, len(perturbation_vector), 2):
                perturbation_vector[j] += 2
        return AttackedText(''.join(candidate))

    
