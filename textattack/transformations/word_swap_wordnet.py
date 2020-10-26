"""
Word Swap by swaping synonyms in WordNet
==========================================================
"""


from nltk.corpus import wordnet

import textattack
from textattack.transformations.word_swap import WordSwap


class WordSwapWordNet(WordSwap):
    """Transforms an input by replacing its words with synonyms provided by
    WordNet."""

    def __init__(self, language="eng"):
        if language not in wordnet.langs():
            raise ValueError(f"Language {language} not one of {wordnet.langs()}")
        self.language = language

    def _get_replacement_words(self, word, random=False):
        """Returns a list containing all possible words with 1 character
        replaced by a homoglyph."""
        synonyms = set()
        for syn in wordnet.synsets(word, lang=self.language):
            for syn_word in syn.lemma_names(lang=self.language):
                if (
                    (syn_word != word)
                    and ("_" not in syn_word)
                    and (textattack.shared.utils.is_one_word(syn_word))
                ):
                    # WordNet can suggest phrases that are joined by '_' but we ignore phrases.
                    synonyms.add(syn_word)
        return list(synonyms)
