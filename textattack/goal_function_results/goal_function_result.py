import torch


class GoalFunctionResult:
    """
    Represents the result of a goal function evaluating a AttackedText object.
    
    Args:
        attacked_text: The sequence that was evaluated.
        output: The display-friendly output.
        succeeded: Whether the goal has been achieved.
        score: A score representing how close the model is to achieving its goal.
    """

    def __init__(self, attacked_text, raw_output, output, succeeded, score):
        self.attacked_text = attacked_text
        self.raw_output = raw_output
        self.output = output
        self.score = score
        self.succeeded = succeeded

        if isinstance(self.raw_output, torch.Tensor):
            self.raw_output = self.raw_output.cpu()

        if isinstance(self.score, torch.Tensor):
            self.score = self.score.item()

        if isinstance(self.succeeded, torch.Tensor):
            self.succeeded = self.succeeded.item()

    def get_text_color_input(self):
        """ A string representing the color this result's changed
            portion should be if it represents the original input.
        """
        raise NotImplementedError()

    def get_text_color_perturbed(self):
        """ A string representing the color this result's changed
            portion should be if it represents the perturbed input.
        """
        raise NotImplementedError()

    def get_colored_output(self, color_method=None):
        """ Returns a string representation of this result's output, colored 
            according to `color_method`.
        """
        raise NotImplementedError()
