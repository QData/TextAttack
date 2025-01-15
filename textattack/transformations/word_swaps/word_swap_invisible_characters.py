"""
Word Swap by Invisible Characters
-------------------------------
"""

import numpy as np

from .word_swap import WordSwap

class WordSwapInvisibleCharacters(WordSwap):

    def __init__(self, random_one=False, **kwargs):
        super().__init__(**kwargs)
        self.invisible_chars = ["\u200B", "\u200C", "\u200D"] 
        self.random_one = random_one

    def _get_replacement_words(self, word):

        print("word:", word)

        candidate_words = []
        word_len = len(word)


        if self.random_one:
            char = np.random.choice(self.invisible_chars)
            pos = np.random.randint(0, word_len + 1)
            candidate_words.append(word[:pos] + char + word[pos:])
        else:
            for char in self.invisible_chars:
                for pos in range(word_len + 1):
                    transformed_word = word[:pos] + char + word[pos:]
                    candidate_words.append(transformed_word)

        print("candidate_words:", candidate_words)

        return candidate_words

    @property
    def deterministic(self):
        return not self.random_one
