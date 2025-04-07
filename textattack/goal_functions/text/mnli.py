import functools
import textattack
from .text_to_text_goal_function import TextToTextGoalFunction
import numpy as np

class Mnli(TextToTextGoalFunction):
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

        # emotion_classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

        # print(model_output)

        # predicts = model_output[0]
        # score = predicts[self.ground_truth_output]['score']

        # if np.argmax(list(map(lambda x: x['score'], predicts))) == self.ground_truth_output:
        #     score += 1
        
        # return -score

        # self.ground_truth_output is the target label in int

        return -model_output[self.ground_truth_output]

    
        

    # def extra_repr_keys(self):
    #     if self.maximizable:
    #         return ["maximizable"]
    #     else:
    #         return ["maximizable", "target_distance"]
