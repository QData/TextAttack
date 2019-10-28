import numpy as np

from textattack.constraints import Constraint
from .alzantot_goog_lm import GoogLMHelper

class GoogleLanguageModel(Constraint):
    """ Constraint that uses the Google 1 Billion Words Language Model to 
        determine the difference in perplexity between x and x_adv. 
        
        Returns the top_n sentences.
        
        @TODO allow user to set perplexity threshold; implement __call__.
        
        @TODO this use of the language model only really makes sense for 
            adversarial examples based on word swaps
    """
    def __init__(self, top_n=10):
        self.lm = GoogLMHelper()
    
    def call_many(self, x, x_adv_list):
        """ Returns the `top_n` of x_adv_list, as evaluated by the language 
            model. 
        """
        if not len(x_adv_list): return []
        
        def get_probs(x, x_adv_list):
            word_swap_index = x.first_word_diff_index(x_adv_list[0])
            prefix = x.text_until_word_index(word_swap_index)
            suffix = x.text_after_word_index(word_swap_index)
            swapped_words = np.array([t.words()[word_swap_index] for t in x_adv_list])
            print(' self.lm.get_words_probs(')
            print(prefix)
            print(swapped_words)
            print(suffix)
            print(')')
            probs = self.lm.get_words_probs(prefix, swapped_words, suffix)
            import pdb; pdb.set_trace()
            return probs
        
        word_swap_index_map = {}
        
        for idx, x_adv in enumerate(x_adv_list):
            word_swap_index = x.first_word_diff_index(x_adv)
            if word_swap_index not in word_swap_index_map:
                word_swap_index_map[word_swap_index] = []
            word_swap_index_map[word_swap_index].append((idx, x_adv))
        
        probs = []
        for word_swap_index, (idx, this_x_adv_list) in word_swap_index_map.items():
            probs.append((idx, get_probs(x, this_x_adv_list)))
        
        # Probs is a list of (index, prob) where index is the corresponding 
        # position in x_adv_list.
        print('probs:', probs)
        probs.sort(key=lambda x: x[0])
        
        # Now that they're in order, reduce to just a list of probabilities.
        probs = list(map(lambda x:x[1], probs))
        
        # Get the indices of the maximum elements.
        # @TODO should this be probs or -probs?
        max_el_indices = np.argsort(probs)[:self.top_n]
        
        print('max_el_indices:', max_el_indices)
        
        return np.array(x_adv_list)[max_el_indices]
    
    # def __call__(self, x, x_adv):