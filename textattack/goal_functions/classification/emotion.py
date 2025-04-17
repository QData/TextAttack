import functools
import textattack
from .targeted_classification import TargetedClassification
import numpy as np

class Emotion(TargetedClassification):
    """This is a targeted attack on a sentiment analysis model.
    """

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()

    def _is_goal_complete(self, model_output, _):
        """
        
        """
        return False

    def _get_score(self, model_output, _):
        """
        emotion_classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        ground_truth_output stores the target class' index (int) in the above list (0 to 5).

        model_output is a tensor of probabilities, one for each emotion_class.

        When used with ImperceptibleDE, this method maximises the probability that the model 
        emits for the target class, adding a bonus of 1 if the target class has the highest probability
        emitted by the model among all classes. 
        """

        if np.argmax(model_output) == self.ground_truth_output:
            return -model_output[self.ground_truth_output] - 1
        
        return -model_output[self.ground_truth_output]

    def _get_displayed_output(self, raw_output):
        return raw_output.tolist()
