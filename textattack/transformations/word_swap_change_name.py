from flair.data import Sentence
from flair.models import SequenceTagger
import numpy as np

from textattack.shared.data import PERSON_NAMES
from textattack.transformations import WordSwap


class WordSwapChangeName(WordSwap):
    def __init__(
        self, n=3, first_only=False, last_only=False, confidence_score=0.7, **kwargs
    ):
        """Transforms an input by replacing names of recognized name entity.

        :param n: Number of new names to generate
        :param first_only: Whether to change first name only
        :param last_only: Whether to change last name only
        :param confidence_score: Name will only be changed when it's above confidence score
        """
        super().__init__(**kwargs)
        self.n = n
        self.first_only = first_only
        self.last_only = last_only
        self.confidence_score = confidence_score

    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []

        for i in indices_to_modify:
            word_to_replace = current_text.words[i]
            word_to_replace_ner = current_text.ner_of_word_index(i)
            replacement_words = self._get_replacement_words(
                word_to_replace, word_to_replace_ner
            )
            for r in replacement_words:
                transformed_texts.append(current_text.replace_word_at_index(i, r))

        # print(transformed_texts)

        return transformed_texts

    def _get_replacement_words(self, word, word_part_of_speech):

        replacement_words = []
        tag = word_part_of_speech
        if tag.value == "B-PER" and tag.score > self.confidence_score:
            replacement_words = self._get_firstname(word)
        elif tag.value == "E-PER" and tag.score > self.confidence_score:
            replacement_words = self._get_lastname(word)
        return replacement_words

    def _get_lastname(self, word):
        """Return a list of random last names."""
        return np.random.choice(PERSON_NAMES["last"], self.n)

    def _get_firstname(self, word):
        """Return a list of random first names."""
        return np.random.choice(PERSON_NAMES["first"], self.n)
