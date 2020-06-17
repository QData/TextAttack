from nltk.corpus import wordnet

from textattack.transformations.word_swap import WordSwap


class WordSwapWordNet(WordSwap):
    """ Transforms an input by replacing its words with synonyms provided by WordNet.
    """

    def _get_replacement_words(self, word, random=False):
        """ Returns a list containing all possible words with 1 character replaced by a homoglyph.
        """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                if l.name() != word and check_if_one_word(l.name()):
                    # WordNet can suggest phrases that are joined by '_' but we ignore phrases.
                    synonyms.add(l.name())
        return list(synonyms)


def check_if_one_word(word):
    for c in word:
        if not c.isalpha():
            return False
    return True
