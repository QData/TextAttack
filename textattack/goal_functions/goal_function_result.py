from textattack.shared import utils

class GoalFunctionResult:
    """
    Represents the result of a goal function evaluating a TokenizedText object.
    Args:
        tokenized_text: The sequence that was evaluated.
        output: The display-friendly output.
        succeeded: Whether the goal has been achieved.
        score: A score representing how close the model is to achieving its goal.
    """
    def __init__(self, tokenized_text, output, succeeded, score):
        self.tokenized_text = tokenized_text
        self.output = output
        self.score = score
        self.succeeded = succeeded
    
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
        """ Returns this results output, colored according to `color_method`.
        """
        return utils.color_from_label(self.output)