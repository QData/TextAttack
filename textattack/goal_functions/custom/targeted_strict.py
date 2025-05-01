"""

Goal Function for Strict targeted classification
-------------------------------------------------------
"""

from textattack.goal_functions import GoalFunction
from textattack.goal_function_results import TargetedStrictGoalFunctionResult
import numpy as np
import torch

class TargetedStrict(GoalFunction):
    """A modified targeted attack on classification models which only sets _is_goal_complete to True if argmax(model_output) matches the target_class.
    In TargetedClassification, if either argmax(model_output) == target_class or ground_truth_output == target_class, then _is_goal_complete returns True.
    """

    def __init__(self, *args, target_class=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_class = target_class

    def _process_model_outputs(self, inputs, scores):
        """Processes and validates a list of model outputs.

        This is a task-dependent operation. For example, classification
        outputs need to have a softmax applied.
        """
        # Automatically cast a list or ndarray of predictions to a tensor.
        if isinstance(scores, list) or isinstance(scores, np.ndarray):
            scores = torch.tensor(scores)

        # Ensure the returned value is now a tensor.
        if not isinstance(scores, torch.Tensor):
            raise TypeError(
                "Must have list, np.ndarray, or torch.Tensor of "
                f"scores. Got type {type(scores)}"
            )

        # Validation check on model score dimensions
        if scores.ndim == 1:
            # Unsqueeze prediction, if it's been squeezed by the model.
            if len(inputs) == 1:
                scores = scores.unsqueeze(dim=0)
            else:
                raise ValueError(
                    f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
                )
        elif scores.ndim != 2:
            # If model somehow returns too may dimensions, throw an error.
            raise ValueError(
                f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
            )
        elif scores.shape[0] != len(inputs):
            # If model returns an incorrect number of scores, throw an error.
            raise ValueError(
                f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
            )
        elif not ((scores.sum(dim=1) - 1).abs() < 1e-6).all():
            # Values in each row should sum up to 1. The model should return a
            # set of numbers corresponding to probabilities, which should add
            # up to 1. Since they are `torch.float` values, allow a small
            # error in the summation.
            scores = torch.nn.functional.softmax(scores, dim=1)
            if not ((scores.sum(dim=1) - 1).abs() < 1e-6).all():
                raise ValueError("Model scores do not add up to 1.")
        return scores.cpu()

    def _is_goal_complete(self, model_output, _):
        return self.target_class == model_output.argmax()

    def _get_score(self, model_output, _):
        return model_output[self.target_class]

    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        return TargetedStrictGoalFunctionResult