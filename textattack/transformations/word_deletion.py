from textattack.transformations.word_swap import WordSwap

class WordDeletion(WordSwap):
    """ Transforms an input by deleting a word.
    """

    def _get_replacement_words(self, _):
        """ Returns a list containing all possible word replacements.
        """
        return ['']
    
    def extra_repr_keys(self): 
        return super().extra_repr_keys()
