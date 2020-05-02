from textattack.attack_results import AttackResult
from textattack.shared import utils

class SkippedAttackResult(AttackResult):
    def __init__(self, original_text, original_output):
        super().__init__(original_text, original_text, original_output, original_output)

    def __data__(self, color_method=None):
        data = (self.result_str(color_method), self.original_text.text)
        return tuple(map(str, data))

    def result_str(self, color_method=None):
        failed_str = utils.color_label('[SKIPPED]', 'gray', color_method)
        return utils.color_label(self.original_output, method=color_method) + '-->' + failed_str 
