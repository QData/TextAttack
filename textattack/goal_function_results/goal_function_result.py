"""

goal function result class
====================================

"""

from abc import ABC, abstractmethod

import torch


class GoalFunctionResultStatus:
    SUCCEEDED = 0
    SEARCHING = 1  # In process of searching for a success
    MAXIMIZING = 2
    SKIPPED = 3


class GoalFunctionResult(ABC):
    """Represents the result of a goal function evaluating a AttackedText
    object.

    Args:
        attacked_text: The sequence that was evaluated.
        output: The display-friendly output.
        goal_status: The ``GoalFunctionResultStatus`` representing the status of the achievement of the goal.
        score: A score representing how close the model is to achieving its goal.
        num_queries: How many model queries have been used
        ground_truth_output: The ground truth output
    """

    def __init__(
        self,
        attacked_text,
        raw_output,
        output,
        goal_status,
        score,
        num_queries,
        ground_truth_output,
    ):
        self.attacked_text = attacked_text
        self.raw_output = raw_output
        self.output = output
        self.score = score
        self.goal_status = goal_status
        self.num_queries = num_queries
        self.ground_truth_output = ground_truth_output

        if isinstance(self.raw_output, torch.Tensor):
            self.raw_output = self.raw_output.cpu()

        if isinstance(self.score, torch.Tensor):
            self.score = self.score.item()

    @abstractmethod
    def get_text_color_input(self):
        """A string representing the color this result's changed portion should
        be if it represents the original input."""
        raise NotImplementedError()

    @abstractmethod
    def get_text_color_perturbed(self):
        """A string representing the color this result's changed portion should
        be if it represents the perturbed input."""
        raise NotImplementedError()

    @abstractmethod
    def get_colored_output(self, color_method=None):
        """Returns a string representation of this result's output, colored
        according to `color_method`."""
        raise NotImplementedError()
