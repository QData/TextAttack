import torch

import textattack
from textattack.shared import utils

from .goal_function_result import GoalFunctionResult


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
        output_label = self.raw_output.argmax()
        confidence_score = self.raw_output[output_label]
        if isinstance(confidence_score, torch.Tensor):
            confidence_score = confidence_score.item()
        output = self.output
        if self.attacked_text.attack_attrs.get("label_names"):
            output = self.attacked_text.attack_attrs["label_names"][output]
            output = textattack.shared.utils.process_label_name(output)
            color = textattack.shared.utils.color_from_output(output, output_label)
        else:
            color = textattack.shared.utils.color_from_label(output_label)
        # concatenate with label and convert confidence score to percent, like '33%'
        output_str = f"{output} ({confidence_score:.0%})"
        return utils.color_text(output_str, color=color, method=color_method)
