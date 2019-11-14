import numpy as np
import os

from textattack import utils as utils
from textattack.transformations.word_swap import WordSwap

class WordSwapNeighboringCharacterSwap(WordSwap):
    """ Transforms an input by replacing its words with a neighboring character swap.
    """

    def __init__(self, replace_stopwords=False):
        super().__init__(replace_stopwords)

    def _get_replacement_words(self, word, random=False, max_candidates=15):
        """ If random, returns a list with a single candidate word where 1 pair of neighboring characters is swapped
        If not random, returns a list containing all possible words with 1 pair of neighboring characters swapped
        """
        candidate_words = []

        if len(word) == 1:
            return candidate_words

        if random:
            i = np.random.randint(0, len(word)-1)
            candidate_word = word[:i]+word[i+1]+word[i]+word[i+2:]
            candidate_words.append(candidate_word)
        else:
            for i in range(len(word)-1):
                candidate_word = word[:i]+word[i+1]+word[i]+word[i+2:]
                candidate_words.append(candidate_word)
                if len(candidate_words) > max_candidates:
                    break

        return candidate_words