"""

TextToTextGoalFunctionResult Class
====================================

text2text goal function Result

"""

from .goal_function_result import GoalFunctionResult


class TextToTextGoalFunctionResult(GoalFunctionResult):
    """Represents the result of a text-to-text goal function."""

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
        super().__init__(
            attacked_text,
            raw_output,
            output,
            goal_status,
            score,
            num_queries,
            ground_truth_output,
            goal_function_result_type="Text to Text",
        )

    def get_text_color_input(self):
        """A string representing the color this result's changed portion should
        be if it represents the original input."""
        return "red"

    def get_text_color_perturbed(self):
        """A string representing the color this result's changed portion should
        be if it represents the perturbed input."""
        return "blue"

    def get_colored_output(self, color_method=None):
        """Returns a string representation of this result's output, colored
        according to `color_method`."""
        return str(self.output)
