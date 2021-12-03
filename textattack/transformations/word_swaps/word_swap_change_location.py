"""
Word Swap by Changing Location
-------------------------------
"""
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
    def __init__(self, n=3, confidence_score=0.7, language="en", **kwargs):
        """Transformation that changes recognized locations of a sentence to
        another location that is given in the location map.

        :param n: Number of new locations to generate
        :param confidence_score: Location will only be changed if it's above the confidence score

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

        transformed_texts = []
        for location in location_words:
            idx = location[0]
            word = location[1].capitalize()
            replacement_words = self._get_new_location(word)
            for r in replacement_words:
                if r == word:
                    continue
                text = current_text

                # if original location is more than a single word, remain only the starting word
                if len(idx) > 1:
                    index = idx[1]
                    for i in idx[1:]:
                        text = text.delete_word_at_index(index)

                # replace the starting word with new location
                text = text.replace_word_at_index(idx[0], r)

                transformed_texts.append(text)
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
