from flair.data import Sentence
from flair.models import SequenceTagger
import numpy as np

from textattack.shared.data import PERSON_NAMES
from textattack.transformations import Transformation


class WordSwapChangeName(Transformation):
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
        # really want to silent this line:
        tagger = SequenceTagger.load("ner")

        # TODO: move ner recognition to AttackedText
        sentence = Sentence(current_text.text)
        tagger.predict(sentence)
        fir_name_idx = []
        last_name_idx = []

        # use flair to screen for actual names and eliminate false-positives
        for token in sentence:
            tag = token.get_tag("ner")
            if tag.value == "B-PER" and tag.score > self.confidence_score:
                fir_name_idx.append(token.idx - 1)
            elif tag.value == "E-PER" and tag.score > self.confidence_score:
                last_name_idx.append(token.idx - 1)

        words = current_text.words
        transformed_texts = []

        for i in indices_to_modify:
            word_to_replace = words[i]

            # search for first name replacement
            if i in fir_name_idx and not self.last_only:
                replacement_words = self._get_firstname(word_to_replace)
                transformed_texts_idx = []
                for r in replacement_words:
                    if r == word_to_replace:
                        continue
                    transformed_texts_idx.append(
                        current_text.replace_word_at_index(i, r)
                    )
                transformed_texts.extend(transformed_texts_idx)

            # search for last name replacement
            elif i in last_name_idx and not self.first_only:
                replacement_words = self._get_lastname(word_to_replace)
                transformed_texts_idx = []
                for r in replacement_words:
                    if r == word_to_replace:
                        continue
                    transformed_texts_idx.append(
                        current_text.replace_word_at_index(i, r)
                    )
                transformed_texts.extend(transformed_texts_idx)
        return transformed_texts

    def _get_lastname(self, word):
        """Return a list of random last names."""
        return np.random.choice(PERSON_NAMES["last"], self.n)

    def _get_firstname(self, word):
        """Return a list of random first names."""
        return np.random.choice(PERSON_NAMES["first"], self.n)
