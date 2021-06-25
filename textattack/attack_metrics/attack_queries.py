from .attack_metrics import AttackMetric

class AttackQueries(AttackMetric):
	def __init__(results):
		self.results = results

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