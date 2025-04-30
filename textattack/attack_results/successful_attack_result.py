"""
SuccessfulAttackResult Class
==============================

"""

from .attack_result import AttackResult


class SuccessfulAttackResult(AttackResult):
    """The result of a successful attack."""

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