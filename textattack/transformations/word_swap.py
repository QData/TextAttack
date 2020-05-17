import nltk
from nltk.corpus import stopwords
import random
import string

from .transformation import Transformation

class WordSwap(Transformation):
    """
    An abstract class that takes a sentence and transforms it by replacing
    some of its words.

    Other classes can achieve this by inheriting from WordSwap and 
    overriding self._get_replacement_words.

    Args:
        replace_stopwords(:obj:`bool`, optional): Whether to replace stopwords. Defaults to False. 

    """

    def _get_replacement_words(self, word):
        raise NotImplementedError()
    
    def _get_random_letter(self):
        """ Helper function that returns a random single letter from the English
            alphabet that could be lowercase or uppercase. """
        return random.choice(string.ascii_letters)

    def _get_transformations(self, tokenized_text, modifiable_indices):
        """
        Returns a list of all possible transformations for `text`.
            
        If indices_to_replace is set, only replaces words at those indices.
        
        """
        words = tokenized_text.words
        transformations = []
        word_swaps = []
        for i in modifiable_indices:
            word_to_replace = words[i]
            replacement_words = self._get_replacement_words(word_to_replace)
            new_tokenized_texts = []
            for r in replacement_words:
                # Don't replace with numbers, punctuation, or other non-letter characters.
                if not is_word(r):
                    continue
                new_tokenized_texts.append(tokenized_text.replace_word_at_index(i, r))
            transformations.extend(new_tokenized_texts)
        
        return transformations

def is_word(s):
    """ String `s` counts as a word if it has at least one letter. """
    for c in s:
        if c.isalpha(): return True
    return False 
