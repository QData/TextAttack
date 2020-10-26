"""
Word Swap by swaps characters with QWERTY adjacent keys
==========================================================
"""

import random

from textattack.transformations.word_swap import WordSwap


class WordSwapQWERTY(WordSwap):
    def __init__(
        self, random_one=True, skip_first_char=False, skip_last_char=False, **kwargs
    ):
        """A transformation that swaps characters with adjacent keys on a
        QWERTY keyboard, replicating the kind of errors that come from typing
        too quickly.

        :param random_one: Whether to return a single (random) swap, or all possible swaps.
        :param skip_first_char: When True, do not modify the first character of each word.
        :param skip_last_char: When True, do not modify the last character of each word.
        """
        super().__init__(**kwargs)
        self.random_one = random_one
        self.skip_first_char = skip_first_char
        self.skip_last_char = skip_last_char

        self._keyboard_adjacency = {
            "q": [
                "w",
                "a",
                "s",
            ],
            "w": ["q", "e", "a", "s", "d"],
            "e": ["w", "s", "d", "f", "r"],
            "r": ["e", "d", "f", "g", "t"],
            "t": ["r", "f", "g", "h", "y"],
            "y": ["t", "g", "h", "j", "u"],
            "u": ["y", "h", "j", "k", "i"],
            "i": ["u", "j", "k", "l", "o"],
            "o": ["i", "k", "l", "p"],
            "p": ["o", "l"],
            "a": ["q", "w", "s", "z", "x"],
            "s": ["q", "w", "e", "a", "d", "z", "x"],
            "d": ["w", "e", "r", "f", "c", "x", "s"],
            "f": ["e", "r", "t", "g", "v", "c", "d"],
            "g": ["r", "t", "y", "h", "b", "v", "d"],
            "h": ["t", "y", "u", "g", "j", "b", "n"],
            "j": ["y", "u", "i", "k", "m", "n", "h"],
            "k": ["u", "i", "o", "l", "m", "j"],
            "l": ["i", "o", "p", "k"],
            "z": ["a", "s", "x"],
            "x": ["s", "d", "z", "c"],
            "c": ["x", "d", "f", "v"],
            "v": ["c", "f", "g", "b"],
            "b": ["v", "g", "h", "n"],
            "n": ["b", "h", "j", "m"],
            "m": ["n", "j", "k"],
        }

    def _get_adjacent(self, s):
        s_lower = s.lower()
        if s_lower in self._keyboard_adjacency:
            adjacent_keys = self._keyboard_adjacency[s_lower]
            if s.isupper():
                return [key.upper() for key in adjacent_keys]
            else:
                return adjacent_keys
        else:
            return []

    def _get_replacement_words(self, word):
        if len(word) <= 1:
            return []

        candidate_words = []

        start_idx = 1 if self.skip_first_char else 0
        end_idx = len(word) - (1 + self.skip_last_char)

        if start_idx >= end_idx:
            return []

        if self.random_one:
            i = random.randrange(start_idx, end_idx + 1)
            candidate_word = (
                word[:i] + random.choice(self._get_adjacent(word[i])) + word[i + 1 :]
            )
            candidate_words.append(candidate_word)
        else:
            for i in range(start_idx, end_idx + 1):
                for swap_key in self._get_adjacent(word[i]):
                    candidate_word = word[:i] + swap_key + word[i + 1 :]
                    candidate_words.append(candidate_word)

        return candidate_words

    @property
    def deterministic(self):
        return not self.random_one
