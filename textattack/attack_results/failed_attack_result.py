"""
FailedAttackResult  Class
===========================

"""

from textattack.shared import utils

from .attack_result import AttackResult


class FailedAttackResult(AttackResult):
    """The result of a failed attack."""

    def __init__(self, original_result, perturbed_result=None):
        perturbed_result = perturbed_result or original_result
        super().__init__(original_result, perturbed_result)

    def str_lines(self, color_method=None):
        lines = (
            "[Ground Truth Output] " + self.goal_function_result_str(color_method),
            "[Original Input] " + self.original_text(color_method),
            "[Perturbed Input] " + self.perturbed_result.attacked_text.text,
            "[Perturbed Output] " + str(self.perturbed_result.output)
        )
        return tuple(map(str, lines))

    def goal_function_result_str(self, color_method=None):
        failed_str = utils.color_text("[FAILED]", "red", color_method)
        return (
            self.original_result.get_colored_output(color_method) + " --> " + failed_str
        )
