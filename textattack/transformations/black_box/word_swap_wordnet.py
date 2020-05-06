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
                synonyms.add(l.name()) 
        return list(synonyms)
