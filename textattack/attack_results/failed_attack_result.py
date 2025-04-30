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
            "[Ground Truth Output] " + str(self.original_result.ground_truth_output),
            "[Original Input] " + self.original_text(color_method),
            "[Original Output] " + str(self.original_result.output),
            "[Original Score] " + str(self.original_result.score),
            "[Perturbed Input] " + self.perturbed_result.attacked_text.text,
            "[Perturbed Output] " + str(self.perturbed_result.output),
            "[Perturbed Score] " + str(self.perturbed_result.score)
        )
        return tuple(map(str, lines))

    def goal_function_result_str(self, color_method=None):
        failed_str = utils.color_text("[FAILED]", "red", color_method)
        return (
            self.original_result.get_colored_output(color_method) + " --> " + failed_str
        )
