"""

Goal Function for Logit sum
-------------------------------------------------------
"""

from textattack.goal_functions import GoalFunction
from textattack.goal_function_results import LogitSumGoalFunctionResult
import torch
import numpy as np

class LogitSum(GoalFunction):
    """
    A goal function that minimizes the sum of output logits for classification models.

    This can be used for tasks where the objective is to suppress the model's overall confidence,
    or specifically the logit of the most probable label.

    Behavior:
        - If `target_logit_sum` is set, the attack succeeds when the sum of all logits
          is less than `target_logit_sum`.
        - If `first_element_threshold` is set (or defaulted to 0.5), the attack succeeds
          when the first logit's value is less than that threshold.

    Args:
        target_logit_sum (float, optional): A threshold for the total sum of logits.
        first_element_threshold (float, optional): A fallback threshold for the first logit only.

    Note:
        Only one of `target_logit_sum` or `first_element_threshold` may be set.
    """

    def __init__(self, *args, target_logit_sum=None, first_element_threshold=None, **kwargs):
        """
        Initializes the LogitSum goal function.

        This goal function is used to reduce the model's overall logit output, either by
        minimizing the sum of all logits or by lowering a specific logit's value.

        Args:
            target_logit_sum (float, optional): If set, the attack is successful when the
                sum of all logits is less than this threshold.
            first_element_threshold (float, optional): If `target_logit_sum` is not set,
                this threshold is used to determine success based on whether the first logit's
                value falls below it. Defaults to 0.5 if not specified.
        """
        if ((target_logit_sum is not None) and (first_element_threshold is not None)):
            raise ValueError("Cannot set both target_logit_sum to True and first_element_threshold!")

        self.target_logit_sum = target_logit_sum
        
        if (target_logit_sum is not None) or (first_element_threshold is not None):
            self.first_element_threshold = first_element_threshold
        else:
            self.first_element_threshold = 0.5 # default

        super().__init__(*args, **kwargs)

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
        return scores.cpu()

    def _is_goal_complete(self, model_output, attacked_text):

        if self.target_logit_sum is not None:
            return sum(model_output) < self.target_logit_sum

        return model_output[0] < self.first_element_threshold

    def _get_score(self, model_output, _):
        """
        model_output is a tensor of logits, one for each label.
        """
        return -sum(model_output)

    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        return LogitSumGoalFunctionResult
