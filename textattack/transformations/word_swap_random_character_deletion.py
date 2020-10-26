"""
Word Swap by Random Character Deletion
==========================================================
"""

import numpy as np

# from textattack.shared import utils
from textattack.transformations.word_swap import WordSwap


class WordSwapRandomCharacterDeletion(WordSwap):
    """Transforms an input by deleting its characters.

    Args:
        random_one (bool): Whether to return a single word with a random
            character deleted. If not, returns all possible options.
        skip_first_char (bool): Whether to disregard deleting the first
            character.
        skip_last_char (bool): Whether to disregard deleting the last
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
        """Returns returns a list containing all possible words with 1 letter
        deleted."""
        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 1 if self.skip_first_char else 0
        end_idx = (len(word) - 1) if self.skip_last_char else len(word)

        if start_idx >= end_idx:
            return []

        if self.random_one:
            i = np.random.randint(start_idx, end_idx)
            candidate_word = word[:i] + word[i + 1 :]
            candidate_words.append(candidate_word)
        else:
            for i in range(start_idx, end_idx):
                candidate_word = word[:i] + word[i + 1 :]
                candidate_words.append(candidate_word)

        return candidate_words

    @property
    def deterministic(self):
        return not self.random_one

    def extra_repr_keys(self):
        return super().extra_repr_keys() + ["random_one"]
