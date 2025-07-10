"""

Goal Function for NamedEntityRecognition
-------------------------------------------------------
"""

from textattack.goal_functions import GoalFunction
from textattack.goal_function_results import NamedEntityRecognitionGoalFunctionResult
import numpy as np
import json

class NamedEntityRecognition(GoalFunction):
    """
    A goal function for attacking named entity recognition (NER) models.

    Expects model outputs to be a list of dictionaries, each containing at least:
        - 'entity': the predicted entity label (e.g., "PER", "ORG")
        - 'score': the confidence score associated with that entity

    The goal is to reduce the total confidence of all entities ending with a specified suffix
    (e.g., "PER" for person names), effectively suppressing target entity types.
    """
    
    def __init__(self, *args, target_suffix: str, **kwargs):
        """
        Initializes a NamedEntityRecognition goal function.

        Args:
            target_suffix (str): The suffix of entity labels to target. 
                Only entities whose label ends with this suffix will contribute
                to the score.
        """
        self.target_suffix = target_suffix
        super().__init__(*args, **kwargs)
        
    def _process_model_outputs(self, inputs, scores):
        return scores

    def _is_goal_complete(self, model_output, _):
        score = self._get_score(model_output, None)
        return (-score < 0)

    def _get_score(self, model_output, _):
        """
        Confidence sum
        """

        predicts = model_output
        score = 0
        for predict in predicts:
            if predict['entity'].endswith(self.target_suffix):
                score += predict['score']
        return score

    def _get_displayed_output(self, raw_output):
        serialisable = [
            {**d, "score": float(d["score"])} for d in raw_output
        ]

        json_str = json.dumps(serialisable, ensure_ascii=False, indent=2)
        return json_str

    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        return NamedEntityRecognitionGoalFunctionResult