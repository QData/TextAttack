"""
Word Swap by Homoglyph
-------------------------------
"""

from typing import List
from textattack.shared import AttackedText

# from textattack.shared import utils
from .word_swap import WordSwap

import requests


class WordSwapHomoglyphSwap(WordSwap):
    """Transforms an input by replacing its words with visually similar words
    using homoglyph swaps.

    >>> from textattack.transformations import WordSwapHomoglyphSwap
    >>> from textattack.augmentation import Augmenter

    >>> transformation = WordSwapHomoglyphSwap()
    >>> augmenter = Augmenter(transformation=transformation)
    >>> s = 'I am fabulous.'
    >>> augmenter.augment(s)
    """

    def __init__(self, random_one=False, **kwargs):
        super().__init__(**kwargs)

        # Intentional homoglyphs
        self.homos = dict()
        # Retrieve Unicode Intentional homoglyph characters
        int_resp = requests.get("https://www.unicode.org/Public/security/latest/intentional.txt", stream=True)
        for line in int_resp.iter_lines():
            if len(line):
                line = line.decode('utf-8-sig')
                if line[0] != '#':
                    line = line.replace("#*", "#")
                    _, line = line.split("#", maxsplit=1)
                    if line[3] not in self.homos:
                        self.homos[line[3]] = []
                    self.homos[line[3]].append(line[7])


        self.random_one = random_one

    def get_glyph_map(self, sentence): # attacked text object to glyph_map
        
        glyph_map = []
        for i, char in enumerate(sentence.text):
            if char in self.homos:
                for replacement in self.homos[char]:
                    # glyph_map.append((i, self.homos[char]))
                    glyph_map.append((i, replacement))
        return glyph_map

    def bounds(self, sentence, max_perturbs):
        glyph_map = self.get_glyph_map(sentence)
        return [(-1, len(glyph_map) - 1)] * max_perturbs

    def _get_replacement_words(self, word):
        candidate_words = []
        # if self.random_one:
        #     i = np.random.randint(0, len(word))
        #     if word[i] in self.homos:
        #         repl_letter = self.homos[word[i]]
        #         candidate_word = word[:i] + repl_letter + word[i + 1 :]
        #         candidate_words.append(candidate_word)
        # else:
        #     for i in range(len(word)):
        #         if word[i] in self.homos:
        #             repl_letter = self.homos[word[i]]
        #             candidate_word = word[:i] + repl_letter + word[i + 1 :]
        #             candidate_words.append(candidate_word)
        return candidate_words

    def natural(self, x: float) -> int:
        """Rounds float to the nearest natural number (positive int)"""
        return max(0, round(float(x)))

    def apply_perturbation(self, sentence, perturbation_vector: List[float], glyph_map): # AttackedText object to AttackedText object
        candidate = list(sentence.text)
        for perturb in map(self.natural, perturbation_vector):
            if (perturb >= 0):
                i, char = glyph_map[perturb]
                candidate[i] = char
        return AttackedText(''.join(candidate))

    @property
    def deterministic(self):
        return not self.random_one

    def extra_repr_keys(self):
        return super().extra_repr_keys()
