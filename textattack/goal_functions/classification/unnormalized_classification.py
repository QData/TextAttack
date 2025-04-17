from .classification_goal_function import ClassificationGoalFunction
import numpy as np
import torch

class UnnormalizedClassification(ClassificationGoalFunction):

    """
    This class is suitable for classification tasks where the outputs do not need to sum to 1.
    The output can be a list of probabilities, logits, or other sensible values.
    For example, in multi-label classification we might only wish to output the pure logits.
    By default, ClassificationGoalFunction applies a softmax normalization to ensure the outputs sum to 1.
    This class overrides _process_model_outputs to remove this step.
    """

    def _process_model_outputs(self, inputs, scores):
        if isinstance(scores, list) or isinstance(scores, np.ndarray):
            scores = torch.tensor(scores)

        if not isinstance(scores, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(scores)}")

        if scores.ndim == 1:
            if len(inputs) == 1:
                scores = scores.unsqueeze(dim=0)
            else:
                raise ValueError(
                    f"Model returned shape {scores.shape} for {len(inputs)} inputs."
                )
        elif scores.ndim != 2:
            raise ValueError(
                f"Expected 2D tensor, got shape {scores.shape}"
            )
        elif scores.shape[0] != len(inputs):
            raise ValueError(
                f"Mismatch: scores shape {scores.shape}, inputs length {len(inputs)}"
            )

        return scores.cpu()
