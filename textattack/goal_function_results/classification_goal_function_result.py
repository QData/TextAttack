from .goal_function_result import GoalFunctionResult
from textattack.shared import utils

class ClassificationGoalFunctionResult(GoalFunctionResult):
    """
    Represents the result of a classification goal function.
    """
    
    def get_text_color_input(self):
        """ A string representing the color this result's changed
            portion should be if it represents the original input.
        """
        return utils.color_from_label(self.output)
    
    def get_text_color_perturbed(self):
        """ A string representing the color this result's changed
            portion should be if it represents the perturbed input.
        """
        return utils.color_from_label(self.output)
    
    def get_colored_output(self, color_method=None):
        """ Returns a string representation of this result's output, colored 
            according to `color_method`.
        """
        color = utils.color_from_label(self.output)
        return utils.color_text(str(self.output), color=color, method=color_method)
            