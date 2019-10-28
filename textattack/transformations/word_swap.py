import numpy as np
from .transformation import Transformation

class WordSwap(Transformation):
    """ An abstract class that takes a sentence and transforms it by replacing
        some of its words.
        
        Other classes can achieve this by inheriting from WordSwap and 
        overriding self._get_replacement_words.
    """
    def _get_replacement_words(self, word):
        raise NotImplementedError()
    
    def __call__(self, tokenized_text, indices_to_replace=None):
        """ Returns a list of all possible transformations for `text`.
            
            If indices_to_replace is set, only replaces words at those
                indices.
        """
        words = tokenized_text.words()
        if not indices_to_replace:
            indices_to_replace = list(range(len(words)))
        
        transformations = []
        word_swaps = []
        for i in indices_to_replace:
            word_to_replace = words[i]
            replacement_words = self._get_replacement_words(word_to_replace)
            new_tokenized_texts = []
            for r in replacement_words:
                new_tokenized_texts.append(tokenized_text.replace_word_at_index(i, r))
            transformations.extend(new_tokenized_texts)
        
        return transformations
