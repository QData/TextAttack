"""
Word Swap by Homoglyph
-------------------------------
"""
import numpy as np
from .word_swap_differential_evolution import WordSwapDifferentialEvolution
from typing import List, Tuple
from textattack.shared import AttackedText
import os


class WordSwapHomoglyphSwap(WordSwapDifferentialEvolution):
    """
    Transforms an input by replacing its words with visually similar words
    using homoglyph swaps.

    Based off of Bad Characters: Imperceptible NLP Attacks (Boucher et al., 2021).
    https://arxiv.org/abs/2106.09898 
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

        # Retrieve Unicode Intentional homoglyph characters
        self.homos_intentional = dict()
        path = os.path.dirname(os.path.abspath(__file__))
        path_list = path.split(os.sep)
        path_list = path_list[:-2]
        path_list.append("shared/intentional_homoglyphs.txt")
        homoglyphs_path = os.sep.join(path_list)
        with open(homoglyphs_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    line = line.replace("#*", "#")
                    try:
                        _, data = line.split("#", maxsplit=1)
                        key = data[3]
                        value = data[7]
                        self.homos_intentional.setdefault(key, []).append(value)
                    except IndexError:
                        continue  # skip malformed lines

    def _get_precomputed(self, current_text: AttackedText) -> List[List[Tuple[int, str]]]:
        return [self._get_glyph_map(current_text)]

    def _get_glyph_map(self, current_text: AttackedText) -> List[Tuple[int, str]]:
        glyph_map = []
        for i, char in enumerate(current_text.text):
            if char in self.homos_intentional:
                for replacement in self.homos_intentional[char]:
                    glyph_map.append((i, replacement))
        return glyph_map

    def _get_bounds(self, current_text: AttackedText, max_perturbs: int, precomputed: List[List[Tuple[int, str]]]) -> List[Tuple[int, int]]:  
        glyph_map = precomputed[0]
        return [(-1, len(glyph_map) - 1)] * max_perturbs

    def _natural(self, x: float) -> int:
        """Helper function that rounds float to the nearest natural number (positive int)"""
        return max(0, round(float(x)))

    def apply_perturbation(self, current_text: AttackedText, perturbation_vector: List[float], precomputed: List[List[Tuple[int, str]]]) -> AttackedText: 
        glyph_map = precomputed[0]
        candidate = list(current_text.text)
        for perturb in map(self._natural, perturbation_vector):
            if (perturb >= 0):
                i, char = glyph_map[perturb]
                candidate[i] = char
        return AttackedText(''.join(candidate))

    def _get_replacement_words(self, word: str) -> List[str]:
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

    @property
    def deterministic(self):
        return not self.random_one

    def extra_repr_keys(self):
        return super().extra_repr_keys()
