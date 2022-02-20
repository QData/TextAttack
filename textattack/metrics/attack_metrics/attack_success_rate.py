"""

Metrics on AttackSuccessRate
---------------------------------------------------------------------

"""

from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.metrics import Metric


class AttackSuccessRate(Metric):
    def __init__(self):
        self.failed_attacks = 0
        self.skipped_attacks = 0
        self.successful_attacks = 0

        self.all_metrics = {}

    def calculate(self, results):
        """Calculates all metrics related to number of succesful, failed and
        skipped results in an attack.

        Args:
            results (``AttackResult`` objects):
                Attack results for each instance in dataset
        """
        self.results = results
        self.total_attacks = len(self.results)

        for i, result in enumerate(self.results):
            if isinstance(result, FailedAttackResult):
                self.failed_attacks += 1
                continue
            elif isinstance(result, SkippedAttackResult):
                self.skipped_attacks += 1
                continue
            else:
                self.successful_attacks += 1

        # Calculated numbers
        self.all_metrics["successful_attacks"] = self.successful_attacks
        self.all_metrics["failed_attacks"] = self.failed_attacks
        self.all_metrics["skipped_attacks"] = self.skipped_attacks

        # Percentages wrt the calculations
        self.all_metrics["original_accuracy"] = self.original_accuracy_perc()
        self.all_metrics["attack_accuracy_perc"] = self.attack_accuracy_perc()
        self.all_metrics["attack_success_rate"] = self.attack_success_rate_perc()

        return self.all_metrics

    def original_accuracy_perc(self):
        original_accuracy = (
            (self.total_attacks - self.skipped_attacks) * 100.0 / (self.total_attacks)
        )
        original_accuracy = round(original_accuracy, 2)
        return original_accuracy

    def attack_accuracy_perc(self):
        accuracy_under_attack = (self.failed_attacks) * 100.0 / (self.total_attacks)
        accuracy_under_attack = round(accuracy_under_attack, 2)
        return accuracy_under_attack

    def attack_success_rate_perc(self):
        if self.successful_attacks + self.failed_attacks == 0:
            attack_success_rate = 0
        else:
            attack_success_rate = (
                self.successful_attacks
                * 100.0
                / (self.successful_attacks + self.failed_attacks)
            )
        attack_success_rate = round(attack_success_rate, 2)
        return attack_success_rate
