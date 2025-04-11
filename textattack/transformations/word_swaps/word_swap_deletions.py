from .word_swap import WordSwap
from typing import List, Tuple
from textattack.shared import AttackedText

class WordSwapDeletions(WordSwap):
    """
    Generates visually similar text transformations by embedding Unicode control characters 
    (e.g., backspace, delete, carriage return).

    Based off of Bad Characters: Imperceptible NLP Attacks (Boucher et al., 2021).
    https://arxiv.org/abs/2106.09898 
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.del_chr = chr(0x8)
        self.ins_chr_min = '!'
        self.ins_chr_max = '~'

    def bounds(self, sentence: AttackedText, max_perturbs: int) -> List[Tuple[int, int]]:
        return [(-1, len(sentence.text) - 1), (ord(self.ins_chr_min), ord(self.ins_chr_max))] * max_perturbs

    def _get_replacement_words(self, word: str) -> List[str]:
        candidate_words = []
        return candidate_words

    def natural(self, x: float) -> int:
        """Rounds float to the nearest natural number (positive int)"""
        return max(0, round(float(x)))
    
    def apply_perturbation(self, sentence: AttackedText, perturbation_vector: List[float]) -> AttackedText: 
        candidate = list(sentence.text)
        for i in range(0, len(perturbation_vector), 2):
            idx = self.natural(perturbation_vector[i])
            char = chr(self.natural(perturbation_vector[i+1]))
            candidate = candidate[:idx] + [char, self.del_chr] + candidate[idx:]
            for j in range(i, len(perturbation_vector), 2):
                perturbation_vector[j] += 2
        return AttackedText(''.join(candidate))

    def _get_replacement_words(self, word: str) -> List[str]:
        candidate_words = []
        for i in range(len(word) + 1):  # +1 to allow insertions at the end too
            for code_point in range(ord(self.ins_chr_min), ord(self.ins_chr_max) + 1):
                insert_char = chr(code_point)
                perturbed = (
                    word[:i] + insert_char + self.del_chr + word[i:]
                )
                candidate_words.append(perturbed)
        return candidate_words

    
