import numpy as np
import os

from textattack import utils as utils
from textattack.transformations.word_swap import WordSwap

class WordSwapHomoglyph(WordSwap):
    """ Transforms an input by replacing its words with visually-similar words using homoglyph swaps.
    """

    def __init__(self, replace_stopwords=False):
        super().__init__(replace_stopwords)
        self.homos = {'-':'˗','9':'৭','8':'Ȣ','7':'𝟕','6':'б','5':'Ƽ','4':'Ꮞ','3':'Ʒ','2':'ᒿ','1':'l','0':'O',"'":'`','a': 'ɑ', 'b': 'Ь', 'c': 'ϲ', 'd': 'ԁ', 'e': 'е', 'f': '𝚏', 'g': 'ɡ', 'h': 'հ', 'i': 'і', 'j': 'ϳ', 'k': '𝒌', 'l': 'ⅼ', 'm': 'ｍ', 'n': 'ո', 'o':'о', 'p': 'р', 'q': 'ԛ', 'r': 'ⲅ', 's': 'ѕ', 't': '𝚝', 'u': 'ս', 'v': 'ѵ', 'w': 'ԝ', 'x': '×', 'y': 'у', 'z': 'ᴢ'}

    def _get_replacement_words(self, word, max_candidates=15):
        """ Returns a list containing all possible words with 1 character replaced by a homoglyph.
        """
        candidate_words = []
        for i in range(len(word)):
            if word[i] in self.homos:
                repl_letter = self.homos[word[i]]
                candidate_word = word[:i] + repl_letter + word[i+1:]
                candidate_words.append(candidate_word)

        return candidate_words