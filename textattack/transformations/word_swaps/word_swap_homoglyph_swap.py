"""
Word Swap by Homoglyph
-------------------------------
"""

from .word_swap import WordSwap
from typing import List, Tuple
from textattack.shared import AttackedText
import requests


class WordSwapHomoglyphSwap(WordSwap):
    """
    Transforms an input by replacing its words with visually similar words
    using homoglyph swaps.

    Based off of Bad Characters: Imperceptible NLP Attacks (Boucher et al., 2021).
    https://arxiv.org/abs/2106.09898 
    """

    def __init__(self, **kwargs):
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

    def get_glyph_map(self, sentence: AttackedText) -> List[Tuple[int, str]]:
        glyph_map = []
        for i, char in enumerate(sentence.text):
            if char in self.homos:
                for replacement in self.homos[char]:
                    glyph_map.append((i, replacement))
        return glyph_map

    def bounds(self, sentence: AttackedText, max_perturbs: int) -> List[Tuple[int, int]]:  
        glyph_map = self.get_glyph_map(sentence)
        return [(-1, len(glyph_map) - 1)] * max_perturbs

    def natural(self, x: float) -> int:
        """Rounds float to the nearest natural number (positive int)"""
        return max(0, round(float(x)))

    def apply_perturbation(self, sentence: AttackedText, perturbation_vector: List[float], glyph_map: List[Tuple[int, str]]) -> AttackedText: 
        candidate = list(sentence.text)
        for perturb in map(self.natural, perturbation_vector):
            if (perturb >= 0):
                i, char = glyph_map[perturb]
                candidate[i] = char
        return AttackedText(''.join(candidate))

    def _get_replacement_words(self, word: str) -> List[str]:
        candidate_words = []
        for i in range(len(word)):
            char = word[i]
            if char in self.homos:
                for replacement in self.homos[char]:
                    candidate_word = word[:i] + replacement + word[i+1:]
                    candidate_words.append(candidate_word)
        return candidate_words
