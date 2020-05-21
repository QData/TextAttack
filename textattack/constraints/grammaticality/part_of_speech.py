import lru
import nltk

from textattack.constraints import Constraint
from textattack.shared import TokenizedText
from textattack.shared.validators import transformation_consists_of_word_swaps

class PartOfSpeech(Constraint):
    """ Constraints word swaps to only swap words with the same part of speech.
        Uses the NLTK universal part-of-speech tagger by default.
        An implementation of `<https://arxiv.org/abs/1907.11932>`_
        adapted from `<https://github.com/jind11/TextFooler>`_. 
    """
    def __init__(self, tagset='universal', allow_verb_noun_swap=True):
        self.tagset = tagset
        self.allow_verb_noun_swap = allow_verb_noun_swap
        self._pos_tag_cache = lru.LRU(2**14)
  
    def _can_replace_pos(self, pos_a, pos_b):
        return (pos_a == pos_b) or (self.allow_verb_noun_swap and set([pos_a,pos_b]) <= set(['NOUN','VERB']))

    def _get_pos(self, before_ctx, word, after_ctx):
        context_words = before_ctx + [word] + after_ctx
        context_key = ' '.join(context_words)
        if context_key in self._pos_tag_cache:
            pos_list = self._pos_tag_cache[context_key]
        else:
            _, pos_list = zip(*nltk.pos_tag(context_words, tagset=self.tagset))
            self._pos_tag_cache[context_key] = pos_list
        return pos_list 
        
    def _check_constraint(self, transformed_text, current_text, original_text=None):
        try:
            indices = transformed_text.attack_attrs['newly_modified_indices']
        except KeyError:
            raise KeyError('Cannot apply part-of-speech constraint without `newly_modified_indices`')
        
        for i in indices:
            current_word = current_text.words[i]
            transformed_word = transformed_text.words[i]
            before_ctx = current_text.words[max(i-4,0):i]
            after_ctx = current_text.words[i+1:min(i+5,len(current_text.words))]
            cur_pos = self._get_pos(before_ctx, current_word, after_ctx)
            replace_pos = self._get_pos(before_ctx, transformed_word, after_ctx)
            if not self._can_replace_pos(cur_pos, replace_pos):
                return False

        return True

    def check_compatibility(self, transformation):
        return transformation_consists_of_word_swaps(transformation)

    def extra_repr_keys(self):
        return ['tagset', 'allow_verb_noun_swap']
