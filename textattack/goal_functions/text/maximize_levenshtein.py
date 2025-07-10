from Levenshtein import distance as levenshtein_distance
from .text_to_text_goal_function import TextToTextGoalFunction

class MaximizeLevenshtein(TextToTextGoalFunction):
    """Attempts to maximise the Levenshtein distance between the current output
    translation and the reference translation.

    Levenshtein distance is defined as the minimum number of single-character
    edits (insertions, deletions, or substitutions) required to change one string into another.
    """

    def __init__(self, *args, target_distance=None, **kwargs):
        self.target_distance = target_distance
        super().__init__(*args, **kwargs)

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()

    def _is_goal_complete(self, model_output, _):
        if self.target_distance:
            distance = -self._get_score(model_output, _)
            return distance >= self.target_distance
        else:
            return False

    def _get_score(self, model_output, _):
        distance = levenshtein_distance(model_output, self.ground_truth_output)

        return -distance

    def extra_repr_keys(self):
        if self.maximizable:
            return ["maximizable"]
        else:
            return ["maximizable", "target_distance"]
