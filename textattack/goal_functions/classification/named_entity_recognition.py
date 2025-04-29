from .classification_goal_function import ClassificationGoalFunction
import numpy as np
import json

class NamedEntityRecognition(ClassificationGoalFunction):
    """
    Needs as model output a list of dictionaries, each containing 'entity' and 'score' fields
    """
    
    def __init__(self, *args, target_suffix: str, **kwargs):
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