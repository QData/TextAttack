import numpy as np

from textattack.shared import utils
from textattack.transformations.word_swap import WordSwap

class WordSwapNeighboringCharacterSwap(WordSwap):
    """ Transforms an input by replacing its words with a neighboring character swap.
            
        Args:
            random_one (bool): Whether to return a single word with two characters 
                swapped. If not, returns all possible options.
    """
    def __init__(self, random_one=True, **kwargs):
        super().__init__(**kwargs)
        self.random_one = random_one

    def _get_replacement_words(self, word):
        """ Returns a list containing all possible words with 1 pair of neighboring characters 
            swapped.
        """

        if len(word) <= 1:
            return []
        
        candidate_words = []

        if self.random_one:
            i = np.random.randint(0, len(word)-1)
            candidate_word = word[:i]+word[i+1]+word[i]+word[i+2:]
            candidate_words.append(candidate_word)
        else:
            for i in range(len(word)-1):
                candidate_word = word[:i]+word[i+1]+word[i]+word[i+2:]
                candidate_words.append(candidate_word)

        return candidate_words
        
    def extra_repr_keys(self): 
        return super().extra_repr_keys() + ['random_one']
