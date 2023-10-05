"""
Word Swap by Changing Location
-------------------------------
"""
from collections import defaultdict

import more_itertools as mit
import numpy as np

from textattack.shared.data import NAMED_ENTITIES

from .word_swap import WordSwap


def idx_to_words(ls, words):
    """Given a list generated from cluster_idx, return a list that contains
    sub-list (the first element being the idx, and the second element being the
    words corresponding to the idx)"""

    output = []
    for sub_ls in ls:
        word = words[sub_ls[0]]
        for idx in sub_ls[1:]:
            word = " ".join([word, words[idx]])
        output.append([sub_ls, word])
    return output


class WordSwapChangeLocation(WordSwap):
    def __init__(
        self, n=3, confidence_score=0.7, language="en", consistent=False, **kwargs
    ):
        """Transformation that changes recognized locations of a sentence to
        another location that is given in the location map.

        :param n: Number of new locations to generate
        :param confidence_score: Location will only be changed if it's above the confidence score
        :param consistent:  Whether to change all instances of the same location to the same new location

        >>> from textattack.transformations import WordSwapChangeLocation
        >>> from textattack.augmentation import Augmenter

        >>> transformation = WordSwapChangeLocation()
        >>> augmenter = Augmenter(transformation=transformation)
        >>> s = 'I am in Dallas.'
        >>> augmenter.augment(s)
        """
        super().__init__(**kwargs)
        self.n = n
        self.confidence_score = confidence_score
        self.language = language
        self.consistent = consistent

    def _get_transformations(self, current_text, indices_to_modify):
        words = current_text.words
        location_idx = []
        if self.language == "en":
            model_name = "ner"
        elif self.language == "fra" or self.language == "french":
            model_name = "flair/ner-french"
        else:
            model_name = "flair/ner-multi-fast"
        for i in indices_to_modify:
            tag = current_text.ner_of_word_index(i, model_name)
            if "LOC" in tag.value and tag.score > self.confidence_score:
                location_idx.append(i)

        # Combine location idx and words to a list ([0] is idx, [1] is location name)
        # For example, [1,2] to [ [1,2] , ["New York"] ]
        location_idx = [list(group) for group in mit.consecutive_groups(location_idx)]
        location_words = idx_to_words(location_idx, words)

        if self.consistent:
            location_to_indices = defaultdict(list)
            for idx, location in location_words:
                location_to_indices[self._capitalize(location)].append(idx[0])

        transformed_texts = []
        for location in location_words:
            idx = location[0]
            word = self._capitalize(location[1])
            replacement_words = self._get_new_location(word)
            for r in replacement_words:
                if r == word:
                    continue

                if self.consistent:
                    # If we're doing consistent replacements, only replace the word
                    # if it hasn't already been replaced in a previous iteration
                    if word not in location_to_indices:
                        continue

                    indices_to_delete = []
                    if len(idx) > 1:
                        for i in location_to_indices[word]:
                            for j in range(1, len(idx)):
                                indices_to_delete.append(i + j)

                    transformed_texts.append(current_text.replace_words_at_indices(
                        location_to_indices[word] + indices_to_delete,
                        ([r] * len(location_to_indices[word]))
                        + ([""] * len(indices_to_delete)),
                    ))

                    # Delete this word to mark it as replaced
                    del location_to_indices[word]
                else:
                    # If the original location is more than a single word, keep only the starting word
                    # and replace the starting word with the new word
                    indices_to_delete = idx[1:]
                    transformed_texts.append(current_text.replace_words_at_indices([idx[0]] + indices_to_delete, [r] + [""] * len(indices_to_delete)))

        return transformed_texts

    def _get_new_location(self, word):
        """Return a list of new locations, with the choice of country,
        nationality, and city."""
        language = ""
        if self.language == "esp" or self.language == "spanish":
            language = "-spanish"
        elif self.language == "fra" or self.language == "french":
            language = "-french"
        if word in NAMED_ENTITIES["country" + language]:
            return np.random.choice(NAMED_ENTITIES["country" + language], self.n)
        elif word in NAMED_ENTITIES["nationality" + language]:
            return np.random.choice(NAMED_ENTITIES["nationality" + language], self.n)
        elif word in NAMED_ENTITIES["city"]:
            return np.random.choice(NAMED_ENTITIES["city"], self.n)
        return []

    def _capitalize(self, string):
        """Capitalizes all words in the string."""
        return " ".join(word.capitalize() for word in string.split())
