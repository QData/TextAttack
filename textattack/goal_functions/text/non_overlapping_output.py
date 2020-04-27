import functools
from textattack.shared.utils import words_from_text
from .text_goal_function import TextGoalFunction

class NonOverlappingOutput(TextGoalFunction):
    """
    Ensures that none of the words at a position are equal.
    
    Defined in seq2sick (https://arxiv.org/pdf/1803.01128.pdf), equation (3).
    """

    def _is_goal_complete(self, model_output, correct_output):
        return word_overlap_score(model_output, correct_output) == 0

    def _get_score(self, model_output, correct_output):
        return -word_overlap_score(model_output, correct_output)

@functools.lru_cache(maxsize=2**12)
def word_overlap_score(s1, s2):
    """ Returns the number of words that overlap between s1 and s2. """
    s1_words = words_from_text(s1)
    s2_words = words_from_text(s2)
    num_non_overlapping_words = abs(len(s1_words) - len(s2_words))
    for i in range(min(len(s1_words), len(s2_words))):
        if s1_words[i] != s2_words[i]:
            num_non_overlapping_words += 1
    return num_non_overlapping_words