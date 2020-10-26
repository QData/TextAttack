"""
Word Swap by Neighboring Character Swap
============================================
"""

import numpy as np

# from textattack.shared import utils
from textattack.transformations.word_swap import WordSwap


class WordSwapNeighboringCharacterSwap(WordSwap):
    """Transforms an input by replacing its words with a neighboring character
    swap.

    Args:
        random_one (bool): Whether to return a single word with two characters
            swapped. If not, returns all possible options.
        skip_first_char (bool): Whether to disregard perturbing the first
            character.
        skip_last_char (bool): Whether to disregard perturbing the last
            character.
    """

    def __init__(
        self, random_one=True, skip_first_char=False, skip_last_char=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.random_one = random_one
        self.skip_first_char = skip_first_char
        self.skip_last_char = skip_last_char

    def _get_replacement_words(self, word):
        """Returns a list containing all possible words with 1 pair of
        neighboring characters swapped."""

        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 1 if self.skip_first_char else 0
        end_idx = (len(word) - 2) if self.skip_last_char else (len(word) - 1)

        if start_idx >= end_idx:
            return []

        if self.random_one:
            i = np.random.randint(start_idx, end_idx)
            candidate_word = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
            candidate_words.append(candidate_word)
        else:
            for i in range(start_idx, end_idx):
                candidate_word = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
                candidate_words.append(candidate_word)

        return candidate_words

    @property
    def deterministic(self):
        return not self.random_one

    def extra_repr_keys(self):
        return super().extra_repr_keys() + ["random_one"]
