import functools
import textattack
from .text_to_text_goal_function import TextToTextGoalFunction
import numpy as np

class Emotion(TextToTextGoalFunction):
    """This is a targeted attack on a sentiment analysis model.
    """

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()

    def _is_goal_complete(self, model_output, _):
        score = self._get_score(model_output, _)
        return False

    def _get_score(self, model_output, _):
        """
        emotion_classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        ground_truth_output stores the target class' index in the above list (0 to 5).
        It maximises the probability that the model emits for the target class, adding a bonus of 1 
        if the target class has the highest probability emitted by the model among all classes. 
        """
        print(model_output)
        predicts = model_output[0]
        score = predicts[self.ground_truth_output]['score']

        if np.argmax(list(map(lambda x: x['score'], predicts))) == self.ground_truth_output:
            score += 1
        
        return -score
