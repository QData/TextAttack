from flair.data import Sentence
from flair.models import SequenceTagger
import numpy as np

from textattack.shared.data import NAMED_ENTITIES
from textattack.transformations import Transformation


def cluster_idx(idx_ls):
    """Given a list of idx, return a list that contains sub-lists of adjacent
    idx."""

    if len(idx_ls) < 2:
        return [[i] for i in idx_ls]
    else:
        output = [[idx_ls[0]]]
        prev = idx_ls[0]
        list_pos = 0

        for idx in idx_ls[1:]:
            if idx - 1 == prev:
                output[list_pos].append(idx)
            else:
                output.append([idx])
                list_pos += 1
            prev = idx
        return output


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


class WordSwapChangeLocation(Transformation):
    def __init__(self, n=3, confidence_score=0.7, **kwargs):
        """Transformation that changes recognized locations of a sentence to
        another location that is given in the location map.

        :param n: Number of new locations to generate
        :param confidence_score: Location will only be changed if it's above the confidence score
        """
        super().__init__(**kwargs)
        self.n = n
        self.confidence_score = confidence_score

    def _get_transformations(self, current_text, indices_to_modify):
        words = current_text.words
        location_idx = []

        for i in indices_to_modify:
            word_to_replace = current_text.words[i]
            tag = current_text.ner_of_word_index(i)
            if "LOC" in tag.value and tag.score > self.confidence_score:
                location_idx.append(i)

        # Combine location idx and words to a list ([0] is idx, [1] is location name)
        # For example, [1,2] to [ [1,2] , ["New York"] ]
        location_idx = cluster_idx(location_idx)
        location_words = idx_to_words(location_idx, words)

        transformed_texts = []
        for location in location_words:
            idx = location[0]
            word = location[1]
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
        print(transformed_texts)
        return transformed_texts

    def _get_new_location(self, word):
        """Return a list of new locations, with the choice of country,
        nationality, and city."""
        if word in NAMED_ENTITIES["country"]:
            return np.random.choice(NAMED_ENTITIES["country"], self.n)
        elif word in NAMED_ENTITIES["nationality"]:
            return np.random.choice(NAMED_ENTITIES["nationality"], self.n)
        elif word in NAMED_ENTITIES["city"]:
            return np.random.choice(NAMED_ENTITIES["city"], self.n)
        return []
