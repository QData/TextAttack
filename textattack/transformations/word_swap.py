import numpy as np
from .transformation import Transformation
from nltk.corpus import stopwords
import nltk

class WordSwap(Transformation):
    """ An abstract class that takes a sentence and transforms it by replacing
        some of its words.
        
        Other classes can achieve this by inheriting from WordSwap and 
        overriding self._get_replacement_words.
    """

    def __init__(self, replace_stopwords=False, check_pos=True, tagset='universal', 
                 allow_verb_noun_swap=True):
        self.replace_stopwords = replace_stopwords
        self.stopwords = set(stopwords.words('english'))
        self.check_pos = check_pos
        self.tagset = tagset
        self.allow_verb_noun_swap = allow_verb_noun_swap

    def _get_replacement_words(self, word):
        raise NotImplementedError()
   
    def _can_sub_pos(self, pos_a, pos_b):
        return pos_a == pos_b or self.allow_verb_noun_swap and set([pos_a,pos_b]) <= set(['NOUN','VERB'])

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
            if not self.replace_stopwords and word_to_replace in self.stopwords:
                continue
            replacement_words = self._get_replacement_words(word_to_replace)
            if self.check_pos:
                _, pos_list = zip(*nltk.pos_tag([word_to_replace] + replacement_words,
                                           tagset=self.tagset))
                replacement_words = [word for idx, word  in enumerate(replacement_words)
                                     if self._can_sub_pos(pos_list[0],pos_list[idx+1])]
            new_tokenized_texts = []
            for r in replacement_words:
                new_tokenized_texts.append(tokenized_text.replace_word_at_index(i, r))
            transformations.extend(new_tokenized_texts)
        
        return transformations
