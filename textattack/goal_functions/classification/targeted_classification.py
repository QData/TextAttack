"""

Determine if an attack has been successful in targeted Classification
-----------------------------------------------------------------------
"""


from .classification_goal_function import ClassificationGoalFunction


class TargetedClassification(ClassificationGoalFunction):
    """A targeted attack on classification models which attempts to maximize
    the score of the target label.

    Complete when the arget label is the predicted label.
    """

    def __init__(self, *args, target_class=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_class = target_class

    def _is_goal_complete(self, model_output, _):
        return (
            self.target_class == model_output.argmax()
        ) or self.ground_truth_output == self.target_class

    def _get_score(self, model_output, _):
        if self.target_class < 0 or self.target_class >= len(model_output):
            raise ValueError(
                f"target class set to {self.target_class} with {len(model_output)} classes."
            )
        else:
            return model_output[self.target_class]

    def extra_repr_keys(self):
        if self.maximizable:
            return ["maximizable", "target_class"]
        else:
            return ["target_class"]
