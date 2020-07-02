from .classification_goal_function import ClassificationGoalFunction


class UntargetedClassification(ClassificationGoalFunction):
    """
    An untargeted attack on classification models which attempts to minimize the 
    score of the correct label until it is no longer the predicted label.
    
    Args:
        target_max_score (int): If set, goal is to reduce model output to
            below this score. Otherwise, goal is to change the overall predicted
            class.
    """

    def __init__(self, *args, target_max_score=None, **kwargs):
        self.target_max_score = target_max_score
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output, ground_truth_output):
        if self.target_max_score:
            return model_output[ground_truth_output] < self.target_max_score
        elif (model_output.numel() == 1) and isinstance(ground_truth_output, float):
            return abs(ground_truth_output - model_output.item()) >= (
                self.target_max_score or 0.5
            )
        else:
            return model_output.argmax() != ground_truth_output

    def _get_score(self, model_output, ground_truth_output):
        # If the model outputs a single number and the ground truth output is
        # a float, we assume that this is a regression task.
        if (model_output.numel() == 1) and isinstance(ground_truth_output, float):
            return abs(model_output.item() - ground_truth_output)
        else:
            return 1 - model_output[ground_truth_output]

    def _get_displayed_output(self, raw_output):
        return int(raw_output.argmax())
