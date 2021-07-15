"""

Max Modification Rate
-----------------------------

"""
import math

from textattack.constraints import PreTransformationConstraint


class MaxModificationRate(PreTransformationConstraint):
    """A constraint that prevents modifying words beyond certain percentage of
    total number of words.

    Args:
        max_rate (:obj:`float`):
            Percentage of words that can be modified. For example, given text of 20 words, `max_rate=0.1` will allow at most 2 words to be modified.
        min_threshold (:obj:`int`, optional, defaults to :obj:`1`):
            The minimum number of words that can be perturbed regardless of `max_rate`. For example, given text of 20 words and `max_rate=0.1`,
            setting`min_threshold=4` will still allow 4 words to be modified even though `max_rate=0.1` only allows 2 words. This is useful since
            text length can vary a lot between samples, and a `N%` modification limit might not make sense for very short text.
    """

    def __init__(self, max_rate, min_threshold=1):
        assert isinstance(max_rate, float), "`max_rate` must be a float."
        assert max_rate >= 0 and max_rate <= 1, "`max_rate` must between 0 and 1."
        assert isinstance(min_threshold, int), "`min_threshold` must an int"

        self.max_rate = max_rate
        self.min_threshold = min_threshold

    def _get_modifiable_indices(self, current_text):
        """Returns the word indices in current_text which are able to be
        modified."""

        threshold = max(
            math.ceil(current_text.num_words * self.max_rate), self.min_threshold
        )
        if len(current_text.attack_attrs["modified_indices"]) >= threshold:
            return set()
        else:
            return set(range(len(current_text.words)))

    def extra_repr_keys(self):
        return ["max_rate", "min_threshold"]
