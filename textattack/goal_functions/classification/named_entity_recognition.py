from .classification_goal_function import ClassificationGoalFunction
import numpy as np
import json

class NamedEntityRecognition(ClassificationGoalFunction):
    """
    A goal function for attacking named entity recognition (NER) models.

    Expects model outputs to be a list of dictionaries, each containing at least:
        - 'entity': the predicted entity label (e.g., "PER", "ORG")
        - 'score': the confidence score associated with that entity

    The goal is to reduce the total confidence of all entities ending with a specified suffix
    (e.g., "PER" for person names), effectively suppressing target entity types.

    Note:
        This goal function cannot be instantiated with `validate_outputs=True`.
    """
    
    def __init__(self, *args, target_suffix: str, **kwargs):
        """
        Initializes a NamedEntityRecognition goal function.

        Args:
            target_suffix (str): The suffix of entity labels to target. 
                Only entities whose label ends with this suffix will contribute
                to the score.

        Keyword Args:
            validate_outputs (bool): Must be False. This goal function expects raw model outputs
                and does not support output validation.

        Raises:
            ValueError: If `validate_outputs` is set to True.
        """
        if kwargs.get("validate_outputs", False) is True:
            raise ValueError("NamedEntityRecognition must be created with validate_outputs=False.")
        super().__init__(*args, validate_outputs=False, **kwargs)
        self.target_suffix = target_suffix

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()

    def _is_goal_complete(self, model_output, _):
        score = self._get_score(model_output)
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