import functools
import textattack
from .targeted_classification import TargetedClassification
import numpy as np

class Mnli(TargetedClassification):
    """This is a targeted attack on mnli models.
    It aims to maximise the probability that the model emits for a given target label.
    """

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()

    def _is_goal_complete(self, model_output, _):
        return False

    def _get_score(self, model_output, _):

        """
        self.ground_truth_output contains the target label (int) based on 
        the label map = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

        model_output is a tensor containing the probabilities for the three classes 
        in the same order as in the label map above (contradiction, neutral, entailment).

        When used with ImperceptibleDE, this method maximises the probability of the target label.
        """

        return -model_output[self.ground_truth_output]

    def _get_displayed_output(self, raw_output):
        return raw_output.tolist()
