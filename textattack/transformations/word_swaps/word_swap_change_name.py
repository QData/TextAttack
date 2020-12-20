"""
Word Swap by Changing Name
============================================
"""

import numpy as np

from textattack.shared.data import PERSON_NAMES

from .word_swap import WordSwap


class WordSwapChangeName(WordSwap):
    def __init__(
        self,
        num_name_replacements=3,
        first_only=False,
        last_only=False,
        confidence_score=0.7,
        **kwargs
    ):
        """Transforms an input by replacing names of recognized name entity.

        :param n: Number of new names to generate per name detected
        :param first_only: Whether to change first name only
        :param last_only: Whether to change last name only
        :param confidence_score: Name will only be changed when it's above confidence score
        """
        super().__init__(**kwargs)
        self.num_name_replacements = num_name_replacements
        if first_only & last_only:
            raise ValueError("first_only and last_only cannot both be true")
        self.first_only = first_only
        self.last_only = last_only
        self.confidence_score = confidence_score

    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []

        for i in indices_to_modify:
            word_to_replace = current_text.words[i].capitalize()
            word_to_replace_ner = current_text.ner_of_word_index(i)
            replacement_words = self._get_replacement_words(
                word_to_replace, word_to_replace_ner
            )
            for r in replacement_words:
                transformed_texts.append(current_text.replace_word_at_index(i, r))

        return transformed_texts

    def _get_replacement_words(self, word, word_part_of_speech):

        replacement_words = []
        tag = word_part_of_speech
        if (
            tag.value in ("B-PER", "S-PER")
            and tag.score >= self.confidence_score
            and not self.last_only
        ):
            replacement_words = self._get_firstname(word)
        elif (
            tag.value in ("E-PER", "S-PER")
            and tag.score >= self.confidence_score
            and not self.first_only
        ):
            replacement_words = self._get_lastname(word)
        return replacement_words

    def _get_lastname(self, word):
        """Return a list of random last names."""
        return np.random.choice(PERSON_NAMES["last"], self.num_name_replacements)

    def _get_firstname(self, word):
        """Return a list of random first names."""
        return np.random.choice(PERSON_NAMES["first"], self.num_name_replacements)
