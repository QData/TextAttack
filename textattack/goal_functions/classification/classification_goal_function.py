"""
Determine for if an attack has been successful in Classification
---------------------------------------------------------------------
"""

import numpy as np
import torch

from textattack.goal_function_results import ClassificationGoalFunctionResult
from textattack.goal_functions import GoalFunction


class ClassificationGoalFunction(GoalFunction):
    """A goal function defined on a model that outputs a probability for some
    number of classes."""

    def __init__(self, *args, validate_outputs=True, display_raw_output=False, **kwargs):
        """
        Args:
            apply_softmax (bool): Whether to apply softmax normalization to model outputs.
            validate_outputs (bool): Whether to validate outputs for correct shape and properties.
        """
        self.validate_outputs = validate_outputs
        self.display_raw_output = display_raw_output
        super().__init__(*args, **kwargs)

    def _process_model_outputs(self, inputs, scores):
        """Processes and validates a list of model outputs.

        This is a task-dependent operation. For example, classification
        outputs need to have a softmax applied.
        """
        if (self.validate_outputs == False):
            return scores

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

    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        return ClassificationGoalFunctionResult

    def extra_repr_keys(self):
        return []

    def _get_displayed_output(self, raw_output):
        if self.display_raw_output:
            return raw_output.tolist()
        return int(raw_output.argmax())