"""
random synonym insertation Transformation
============================================

"""

import random

from nltk.corpus import wordnet

from .word_insertion import WordInsertion


class WordInsertionRandomSynonym(WordInsertion):
    """Transformation that inserts synonyms of words that are already in the
    sequence."""

    def _get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word and check_if_one_word(lemma.name()):
                    synonyms.add(lemma.name())
        return list(synonyms)

    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []
        for idx in indices_to_modify:
            synonyms = []
            # try to find a word with synonyms, and deal with edge case where there aren't any
            for attempt in range(7):
                synonyms = self._get_synonyms(random.choice(current_text.words))
                if synonyms:
                    break
                elif attempt == 6:
                    return [current_text]
            random_synonym = random.choice(synonyms)
            transformed_texts.append(
                current_text.insert_text_before_word_index(idx, random_synonym)
            )
        return transformed_texts

    @property
    def deterministic(self):
        return False


def check_if_one_word(word):
    for c in word:
        if not c.isalpha():
            return False
    return True
