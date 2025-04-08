import functools
import textattack
from .text_to_text_goal_function import TextToTextGoalFunction
import numpy as np

class Mnli(TextToTextGoalFunction):
    """This is a targeted attack on mnli models.
    It aims to maximise the probability that the model emits for a given target label.
    """

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()

    def _is_goal_complete(self, model_output, _):
        score = self._get_score(model_output, _)
        return False

    def _get_score(self, model_output, _):

        # self.ground_truth_output contains the target label (int) based on 
        # the label map = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

        # model_output is a tensor containing the probabilities for the three classes 
        # in the same order as in the label map above (contradiction, neutral, entailment).

        # we return the negative of the probability 

        return -model_output[self.ground_truth_output]
