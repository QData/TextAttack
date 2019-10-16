import numpy as np
from .perturbation import Perturbation

class WordSwap(Perturbation):
    """ An abstract class that takes a sentence and perturbs it by replacing
        some of its words.
        
        Other classes can achieve this by inheriting from WordSwap and 
        overriding self._get_replacement_words.
    """
    def _get_replacement_words(self, word):
        raise NotImplementedError()
    
    def perturb(self, tokenized_text, indices_to_replace=None):
        """ Returns a list of all possible perturbations for `text`.
            
            If indices_to_replace is set, only replaces words at those
                indices.
        """
        words = tokenized_text.words()
        if not indices_to_replace:
            indices_to_replace = list(range(len(words)))
        
        perturbations = []
        word_swaps = []
        for i in indices_to_replace:
            word_to_replace = words[i]
            replacement_words = self._get_replacement_words(word_to_replace)
            new_tokenized_texts = []
            for r in replacement_words:
                new_tokenized_texts.append(tokenized_text.replace_word_at_index(i, r))
            perturbations.extend(new_tokenized_texts)
        
        return self._filter_perturbations(tokenized_text, perturbations)
