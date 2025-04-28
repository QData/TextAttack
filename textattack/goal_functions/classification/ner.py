import functools
import textattack
from .targeted_classification import TargetedClassification
from .unprocessed_classification import UnprocessedClassification
import numpy as np
import json

class Ner(TargetedClassification, UnprocessedClassification):
    """This is a targeted attack on named entity recognition models.
    """

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()

    def _is_goal_complete(self, model_output, _):
        return False

    def _get_score(self, model_output, _):

        """
        model_output is a list of dictionaries, each with keys (entity, score, index, word, start, end)
        example: [{'entity': 'I-MISC', 'score': 0.99509996, 'index': 6, 'word': 'J', 'start': 8, 'end': 9}]
        ground_truth_output stores the target suffix entity we are trying to achieve.
        """

        predicts = model_output
        score = 0
        for predict in predicts:
            if predict['entity'].endswith(self.ground_truth_output):
                score += predict['score']
        return -score

    def _get_displayed_output(self, raw_output):
        serialisable = [
            {**d, "score": float(d["score"])} for d in raw_output
        ]

        json_str = json.dumps(serialisable, ensure_ascii=False, indent=2)
        return json_str