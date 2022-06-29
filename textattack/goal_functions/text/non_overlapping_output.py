"""

Goal Function for seq2sick
-------------------------------------------------------
"""


import functools

import numpy as np

from textattack.shared.utils import words_from_text

from .text_to_text_goal_function import TextToTextGoalFunction


class NonOverlappingOutput(TextToTextGoalFunction):
    """Ensures that none of the words at a position are equal.

    Defined in seq2sick (https://arxiv.org/pdf/1803.01128.pdf), equation
    (3).
    """

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()
        get_words_cached.cache_clear()
        word_difference_score.cache_clear()

    def _is_goal_complete(self, model_output, _):
        return self._get_score(model_output, self.ground_truth_output) == 1.0

    def _get_score(self, model_output, _):
        num_words_diff = word_difference_score(model_output, self.ground_truth_output)
        if num_words_diff == 0:
            return 0.0
        else:
            return num_words_diff / len(get_words_cached(self.ground_truth_output))


@functools.lru_cache(maxsize=2 ** 12)
def get_words_cached(s):
    return np.array(words_from_text(s))


@functools.lru_cache(maxsize=2 ** 12)
def word_difference_score(s1, s2):
    """Returns the number of words that are non-overlapping between s1 and
    s2."""
    s1_words = get_words_cached(s1)
    s2_words = get_words_cached(s2)
    min_length = min(len(s1_words), len(s2_words))
    if min_length == 0:
        return 0
    s1_words = s1_words[:min_length]
    s2_words = s2_words[:min_length]
    return (s1_words != s2_words).sum()
