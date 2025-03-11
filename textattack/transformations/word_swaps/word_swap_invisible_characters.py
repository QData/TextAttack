"""
Word Swap by Invisible Characters
-------------------------------
"""

from typing import List
from .word_swap import WordSwap
from textattack.shared import AttackedText

class WordSwapInvisibleCharacters(WordSwap):

    def __init__(self, random_one=False, **kwargs):
        super().__init__(**kwargs)
        self.invisible_chars = ["\u200B", "\u200C", "\u200D"] 
        self.random_one = random_one

    def bounds(self, sentence, max_perturbs):
        return [(0, len(self.invisible_chars) - 1), (-1, len(sentence.text) - 1)] * max_perturbs

    def _get_replacement_words(self, word):
        candidate_words = []
        # word_len = len(word)
        # if self.random_one:
        #     char = np.random.choice(self.invisible_chars)
        #     pos = np.random.randint(0, word_len + 1)
        #     candidate_words.append(word[:pos] + char + word[pos:])
        # else:
        #     for char in self.invisible_chars:
        #         for pos in range(word_len + 1):
        #             transformed_word = word[:pos] + char + word[pos:]
        #             candidate_words.append(transformed_word)
        return candidate_words

    def natural(self, x: float) -> int:
        return max(0, round(float(x)))

    def apply_perturbation(self, sentence, perturbation_vector: List[float]):
        candidate = list(sentence.text)
        for i in range(0, len(perturbation_vector), 2):
            inp_index = self.natural(perturbation_vector[i+1])
            if (inp_index >= 0):
                inv_char = self.invisible_chars[self.natural(perturbation_vector[i])]
                candidate = candidate[:inp_index] + [inv_char] + candidate[inp_index:]
        return AttackedText(''.join(candidate))

    @property
    def deterministic(self):
        return not self.random_one
