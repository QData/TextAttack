import functools
import textattack
from .targeted_classification import TargetedClassification
import numpy as np

class TargetedStrict(TargetedClassification):

    def _is_goal_complete(self, model_output, _):
        return _get_score(model_output) >= 1

    def _get_score(self, model_output, _):
        if np.argmax(model_output) == self.target_class:
            return model_output[self.target_class] + 1
        
        return model_output[self.target_class]
