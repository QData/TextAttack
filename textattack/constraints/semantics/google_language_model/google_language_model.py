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
        self.top_n = top_n
    
    def call_many(self, x, x_adv_list):
        """ Returns the `top_n` of x_adv_list, as evaluated by the language 
            model. 
        """
        if not len(x_adv_list): return []
        
        def get_probs(x, x_adv_list):
            word_swap_index = x.first_word_diff_index(x_adv_list[0])
            prefix = x.text_until_word_index(word_swap_index)
            swapped_words = np.array([t.words()[word_swap_index] for t in x_adv_list])
            suffix = x.text_after_word_index(word_swap_index)
            print(prefix, swapped_words, suffix)
            probs = self.lm.get_words_probs(prefix, swapped_words, suffix)
            return probs
        
        word_swap_index_map = {}
        
        for idx, x_adv in enumerate(x_adv_list):
            word_swap_index = x.first_word_diff_index(x_adv)
            if word_swap_index not in word_swap_index_map:
                word_swap_index_map[word_swap_index] = []
            word_swap_index_map[word_swap_index].append((idx, x_adv))
        
        probs = []
        for word_swap_index, item_list in word_swap_index_map.items():
            # zip(*some_list) is the inverse operator of zip!
            item_indices, this_x_adv_list = zip(*item_list)
            probs.extend(list(zip(item_indices, get_probs(x, this_x_adv_list))))
        
        # Probs is a list of (index, prob) where index is the corresponding 
        # position in x_adv_list.
        probs.sort(key=lambda x: x[0])
        
        # Now that they're in order, reduce to just a list of probabilities.
        probs = list(map(lambda x:x[1], probs))
        
        # Get the indices of the maximum elements.
        # @TODO should this be probs or -probs?
        max_el_indices = np.argsort(probs)[:self.top_n]
        
        print('max_el_indices:', max_el_indices)
        
        return np.array(x_adv_list)[max_el_indices]
    
    # def __call__(self, x, x_adv):