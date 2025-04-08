import functools
import textattack
from .text_to_text_goal_function import TextToTextGoalFunction
import numpy as np

class Ner(TextToTextGoalFunction):
    """This is a targeted attack on named entity recognition models.
    """

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()

    def _is_goal_complete(self, model_output, _):
        score = self._get_score(model_output, _)
        return False

    def _get_score(self, model_output, _):

        """
        model_output is a list of dictionaries, each with keys (entity, score, index, word, start, end)
        example: [{'entity': 'I-MISC', 'score': 0.99509996, 'index': 6, 'word': 'J', 'start': 8, 'end': 9}]
        ground_truth_output stores the target suffix entity we are trying to achieve
        We aim to maximise the scores of all words for which the model outputs an "entity" value ending in ground_truth_output.
        """

        predicts = model_output
        print(predicts)
        # print(predicts)
        score = 0
        for predict in predicts:
            if predict['entity'].endswith(self.ground_truth_output):
                score += predict['score']
        return -score
