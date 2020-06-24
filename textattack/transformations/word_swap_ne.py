import functools 

import nltk
from nltk.corpus import wordnet

from textattack.transformations.word_swap import WordSwap


class WordSwapNamedEntity(WordSwap):
    """ Transforms an input by replacing a Named Entity (NE) with the most frequently used NE of the same tag type in the dataset.
    """

    @functools.lru_cache(maxsize=2**14)
    def get_entities(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        entities = nltk.chunk.ne_chunk(tagged, binary=True)
        return entities.leaves()

    def _get_replacement_words(self, word, random=False):
        other_ne = set()
        entities = self.get_entities(word)
        # @TODO: grab the most frequently used NE in the dictionary of this same tag type...
        return other_ne

    def check_if_one_word(self, word):
        for c in word:
            if not c.isalpha():
                return False
        return True
