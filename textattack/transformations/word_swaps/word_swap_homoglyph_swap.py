"""
Word Swap by Homoglyph
-------------------------------
"""

import numpy as np
from scipy.optimize import differential_evolution
import random
from typing import List
from textattack.shared import AttackedText

# from textattack.shared import utils
from .word_swap import WordSwap


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
        self.homos = {
            "-": "˗",
            "9": "৭",
            "8": "Ȣ",
            "7": "𝟕",
            "6": "б",
            "5": "Ƽ",
            "4": "Ꮞ",
            "3": "Ʒ",
            "2": "ᒿ",
            "1": "l",
            "0": "O",
            "'": "`",
            "a": "ɑ",
            "b": "Ь",
            "c": "ϲ",
            "d": "ԁ",
            "e": "е",
            "f": "𝚏",
            "g": "ɡ",
            "h": "հ",
            "i": "і",
            "j": "ϳ",
            "k": "𝒌",
            "l": "ⅼ",
            "m": "ｍ",
            "n": "ո",
            "o": "о",
            "p": "р",
            "q": "ԛ",
            "r": "ⲅ",
            "s": "ѕ",
            "t": "𝚝",
            "u": "ս",
            "v": "ѵ",
            "w": "ԝ",
            "x": "×",
            "y": "у",
            "z": "ᴢ",
        }
        self.random_one = random_one

    def get_glyph_map(self, sentence): # attacked text object to glyph_map
        
        glyph_map = []
        for i, char in enumerate(sentence.text):
            if char in self.homos:
                glyph_map.append((i, self.homos[char]))
        return glyph_map

    def bounds(self, sentence, max_perturbs):
        glyph_map = self.get_glyph_map(sentence)
        return [(-1, len(glyph_map) - 1)] * max_perturbs

    def _get_replacement_words(self, word):
        """Returns a list containing all possible words with 1 character
        replaced by a homoglyph."""
        candidate_words = []
        if self.random_one:
            i = np.random.randint(0, len(word))
            if word[i] in self.homos:
                repl_letter = self.homos[word[i]]
                candidate_word = word[:i] + repl_letter + word[i + 1 :]
                candidate_words.append(candidate_word)
        else:
            for i in range(len(word)):
                if word[i] in self.homos:
                    repl_letter = self.homos[word[i]]
                    candidate_word = word[:i] + repl_letter + word[i + 1 :]
                    candidate_words.append(candidate_word)

        return candidate_words

    def natural(self, x: float) -> int:
        """Rounds float to the nearest natural number (positive int)"""
        return max(0, round(float(x)))

    def _apply_perturbation(self, sentence, perturbation_vector: List[float], glyph_map): # AttackedText object to AttackedText object
        print(type(perturbation_vector))
        sentence = list(sentence.text)
        for perturb in map(self.natural, perturbation_vector):
            if (perturb >= 0):
                i, char = glyph_map[perturb]
                sentence[i] = char
        return AttackedText(''.join(sentence))

    @property
    def deterministic(self):
        return not self.random_one

    def extra_repr_keys(self):
        return super().extra_repr_keys()
