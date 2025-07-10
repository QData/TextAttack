"""
Word Swap by Invisible Deletions
----------------------------------
"""

from .word_swap_differential_evolution import WordSwapDifferentialEvolution
from typing import List, Tuple
from textattack.shared import AttackedText
import numpy as np

class WordSwapDeletions(WordSwapDifferentialEvolution):
    """
    Generates visually similar text transformations by embedding Unicode control characters 
    (e.g., backspace, delete, carriage return).

    Based off of Bad Characters: Imperceptible NLP Attacks (Boucher et al., 2021).
    https://arxiv.org/abs/2106.09898 
    """

    def __init__(self, random_one=False, **kwargs):
        super().__init__(**kwargs)
        self.del_chr = chr(0x8)
        self.ins_chr_min = '!'
        self.ins_chr_max = '~'
        self.random_one = random_one

    def _get_bounds(self, current_text: AttackedText, max_perturbs: int, _) -> List[Tuple[int, int]]:
        return [(-1, len(current_text.text) - 1), (ord(self.ins_chr_min), ord(self.ins_chr_max))] * max_perturbs

    def _natural(self, x: float) -> int:
        """Rounds float to the nearest natural number (positive int)"""
        return max(0, round(float(x)))
    
    def apply_perturbation(self, current_text: AttackedText, perturbation_vector: List[float], _) -> AttackedText: 
        candidate = list(current_text.text)
        for i in range(0, len(perturbation_vector), 2):
            idx = self._natural(perturbation_vector[i])
            char = chr(self._natural(perturbation_vector[i+1]))
            candidate = candidate[:idx] + [char, self.del_chr] + candidate[idx:]
            for j in range(i, len(perturbation_vector), 2):
                perturbation_vector[j] += 2
        return AttackedText(''.join(candidate))

    def _get_replacement_words(self, word: str) -> List[str]:
        candidate_words = []
        if self.random_one:
            if len(word) == 0:
                return []
            i = np.random.randint(0, len(word) + 1)
            rand_char = chr(np.random.randint(ord(self.ins_chr_min), ord(self.ins_chr_max) + 1))
            perturbed = word[:i] + rand_char + self.del_chr + word[i:]
            candidate_words.append(perturbed)
        else:
            for i in range(len(word) + 1):  # +1 to allow insertions at the end
                for code_point in range(ord(self.ins_chr_min), ord(self.ins_chr_max) + 1):
                    insert_char = chr(code_point)
                    perturbed = word[:i] + insert_char + self.del_chr + word[i:]
                    candidate_words.append(perturbed)
        return candidate_words

    @property
    def deterministic(self):
        return not self.random_one
    
    def extra_repr_keys(self):
        return super().extra_repr_keys()

    
