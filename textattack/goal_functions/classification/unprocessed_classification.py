from .classification_goal_function import ClassificationGoalFunction
import numpy as np
import torch

class UnprocessedClassification(ClassificationGoalFunction):

    """
    Suitable for special classification tasks like named entity recognition, which classifies
    at the token-level, and therefore removes the requirement of len(scores) == len(inputs).
    """

    def _process_model_outputs(self, inputs, scores):
        return scores
