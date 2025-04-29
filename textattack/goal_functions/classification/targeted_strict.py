from .targeted_classification import TargetedClassification
import numpy as np

class TargetedStrict(TargetedClassification):
    """A modified targeted attack on classification models which awards a bonus score of 1 if the class with the highest predicted probability is exactly equal to the target_class.
    """

    def _is_goal_complete(self, model_output, _):
        return self._get_score(model_output, None) >= 1

    def _get_score(self, model_output, _):
        if np.argmax(model_output) == self.target_class:
            return model_output[self.target_class] + 1
        
        return model_output[self.target_class]
