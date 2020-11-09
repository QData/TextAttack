"""

Min Word Lenth
--------------------------

"""

from textattack.constraints import PreTransformationConstraint


class MinWordLength(PreTransformationConstraint):
    """A constraint that prevents modifications to words less than a certain
    length.

    :param min_length: Minimum length needed for changes to be made to a word.
    """

    def __init__(self, min_length):
        self.min_length = min_length

    def _get_modifiable_indices(self, current_text):
        idxs = []
        for i, word in enumerate(current_text.words):
            if len(word) >= self.min_length:
                idxs.append(i)
        return set(idxs)
