import numpy as np

from textattack.shared import utils
from textattack.transformations.word_swap import WordSwap

class WordSwapRandomCharacterDeletion(WordSwap):
    """ Transforms an input by deleting its characters.
           
        Args:
            random_one (bool): Whether to return a single word with a random 
                character deleted. If not, returns all possible options.
    """
    def __init__(self, random_one=True, **kwargs):
        super().__init__(**kwargs)
        self.random_one = random_one

    def _get_replacement_words(self, word):
        """ Returns returns a list containing all possible words with 1 letter
            deleted.
        """
        if len(word) <= 1:
            return []
            
        candidate_words = []

        if self.random_one:
            i = np.random.randint(0, len(word))
            candidate_word = word[:i] + word[i+1:]
            candidate_words.append(candidate_word)
        else:
            for i in range(len(word)):
                candidate_word = word[:i] + word[i+1:]
                candidate_words.append(candidate_word)

        return candidate_words
        
    def extra_repr_keys(self): 
        return super().extra_repr_keys() + ['random_one']
