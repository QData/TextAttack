import functools
import textattack
from .text_to_text_goal_function import TextToTextGoalFunction

class Toxic(TextToTextGoalFunction):
    """Attempts to minimize the Levenshtein distance between the current output
    translation and the reference translation.

    Levenshtein distance is defined as the minimum number of single-character
    edits (insertions, deletions, or substitutions) required to change one string into another.
    """

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()

    def _is_goal_complete(self, model_output, _):
        score = self._get_score(model_output, _)
        if (score == 0.0):
            return True
        else:
            return False

    def _get_score(self, model_output, _):
        return sum(model_output.values())

    def extra_repr_keys(self):
        if self.maximizable:
            return ["maximizable"]
        else:
            return ["maximizable", "target_distance"]
