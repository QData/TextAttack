"""
Word Swap by Punctuation Character Insertion
------------------------------------------------

"""
import random
import string
from .word_swap import WordSwap
import numpy as np


class WordSwapPunctuationCharacterInsertion(WordSwap):
    """Transforms an input by inserting a random punctuation character.

    random_one (bool): Whether to return a single word with a random
    character inserted. If not, returns all possible options.
    skip_first_char (bool): Whether to disregard inserting as the first
    character. 
    skip_last_char (bool): Whether to disregard inserting as
    the last character.
    >>> from textattack.transformations import WordSwapPunctuationCharacterInsertion
    >>> from textattack.augmentation import Augmenter

    >>> transformation = WordSwapPunctuationCharacterInsertion()
    >>> augmenter = Augmenter(transformation=transformation)
    >>> s = 'I am fabulous.'
    >>> augmenter.augment(s)
    """

    def __init__(
        self, punct=string.punctuation, **kwargs
    ):
        super().__init__(**kwargs)
        self.punct = punct

    def _get_replacement_words(self, word):
        """Returns returns a list containing all possible words with 1 random
        character inserted."""
        return [(word + ' ' + random.choice(self.punct)).strip()]

    @property
    def deterministic(self):
        return False

    def extra_repr_keys(self):
        return super().extra_repr_keys() + ["punct"]
