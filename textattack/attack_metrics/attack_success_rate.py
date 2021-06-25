from .attack_metrics import AttackMetric

class AttackSuccessRate(AttackMetric):
	def __init__(results):
		self.results = results
		self.failed_attacks = 0
        self.skipped_attacks = 0
        self.successful_attacks = 0
        self.total_attacks = len(self.results)

	def calculate():
		for i, self.result in enumerate(self.results):
            if isinstance(result, FailedAttackResult):
                failed_attacks += 1
                continue
            elif isinstance(result, SkippedAttackResult):
                skipped_attacks += 1
                continue
            else:
                successful_attacks += 1

		return self.successful_attacks, self.failed_attacks, self.skipped_attacks


	def original_accuracy():
		original_accuracy = (total_attacks - skipped_attacks) * 100.0 / (total_attacks)
        # original_accuracy = str(round(original_accuracy, 2)) + "%"
        original_accuracy = round(original_accuracy, 2)
		return original_accuracy

	def attack_accuracy():
		accuracy_under_attack = (failed_attacks) * 100.0 / (total_attacks)
        # accuracy_under_attack = str(round(accuracy_under_attack, 2)) + "%"
        accuracy_under_attack = round(accuracy_under_attack, 2)
		return accuracy_under_attack

	def attack_success_rate():
		if successful_attacks + failed_attacks == 0:
            attack_success_rate = 0
        else:
            attack_success_rate = (
                successful_attacks * 100.0 / (successful_attacks + failed_attacks)
            )
        # attack_success_rate = str(round(attack_success_rate, 2)) + "%"
        attack_success_rate = round(attack_success_rate, 2)
		return attack_success_rate