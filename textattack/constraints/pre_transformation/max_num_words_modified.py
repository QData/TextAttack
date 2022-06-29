"""

Max Modification Rate
-----------------------------

"""

from textattack.constraints import PreTransformationConstraint


class MaxNumWordsModified(PreTransformationConstraint):
    def __init__(self, max_num_words: int):
        self.max_num_words = max_num_words

    def _get_modifiable_indices(self, current_text):
        """Returns the word indices in current_text which are able to be
        modified."""

        if len(current_text.attack_attrs["modified_indices"]) >= self.max_num_words:
            return set()
        else:
            return set(range(len(current_text.words)))

    def extra_repr_keys(self):
        return ["max_num_words"]
