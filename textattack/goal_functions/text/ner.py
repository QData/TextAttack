import functools
import textattack
from .text_to_text_goal_function import TextToTextGoalFunction
import numpy as np

class Ner(TextToTextGoalFunction):
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
        return False

    def _get_score(self, model_output, _):

        predicts = model_output
        # print(predicts)
        score = 0
        for predict in predicts:
            if predict['entity'].endswith(self.ground_truth_output):
                score += predict['score']
        return -score

    # def extra_repr_keys(self):
    #     if self.maximizable:
    #         return ["maximizable"]
    #     else:
    #         return ["maximizable", "target_distance"]
