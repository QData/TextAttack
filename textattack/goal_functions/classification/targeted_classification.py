from .classification_goal_function import ClassificationGoalFunction


class TargetedClassification(ClassificationGoalFunction):
    """
    An targeted attack on classification models which attempts to maximize the 
    score of the target label until it is the predicted label.
    """

    def __init__(self, model, target_class=0):
        super().__init__(model)
        self.target_class = target_class

    def _is_goal_complete(self, model_output, ground_truth_output):
        return (
            self.target_class == model_output.argmax()
        ) or ground_truth_output == self.target_class

    def _get_score(self, model_output, _):
        if self.target_class < 0 or self.target_class >= len(model_output):
            raise ValueError(
                f"target class set to {self.target_class} with {len(model_output)} classes."
            )
        else:
            return model_output[self.target_class]

    def _get_displayed_output(self, raw_output):
        return int(raw_output.argmax())

    def extra_repr_keys(self):
        return ["target_class"]
