import torch

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
            
