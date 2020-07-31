from textattack.goal_function_results import TextToTextGoalFunctionResult
from textattack.goal_functions import GoalFunction


class TextToTextGoalFunction(GoalFunction):
    """A goal function defined on a model that outputs text.

    model: The PyTorch or TensorFlow model used for evaluation.
    original_output: the original output of the model
    """

    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        return TextToTextGoalFunctionResult

    def _process_model_outputs(self, _, outputs):
        """Processes and validates a list of model outputs."""
        return outputs.flatten()

    def _get_displayed_output(self, raw_output):
        return raw_output
