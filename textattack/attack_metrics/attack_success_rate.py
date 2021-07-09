from .attack_metric import AttackMetric

from textattack.attack_results import FailedAttackResult, SkippedAttackResult


class AttackSuccessRate(AttackMetric):
	"""Calculates all metrics related to number of succesful, failed and skipped results in an attack

	Args:
	results (:obj::`list`:class:`~textattack.goal_function_results.GoalFunctionResult`):
			Attack results for each instance in dataset
	"""
	def __init__(self,results):
		self.results = results
		self.failed_attacks = 0
		self.skipped_attacks = 0
		self.successful_attacks = 0
		self.total_attacks = len(self.results)

		self.calculate()

	def calculate(self):
		for i, result in enumerate(self.results):
			if isinstance(result, FailedAttackResult):
				self.failed_attacks += 1
				continue
			elif isinstance(result, SkippedAttackResult):
				self.skipped_attacks += 1
				continue
			else:
				self.successful_attacks += 1

	def successful_attacks_num(self):
		return self.successful_attacks

	def failed_attacks_num(self):	
		return self.failed_attacks 

	def skipped_attacks_num(self): 
		return self.skipped_attacks

	def original_accuracy_perc(self):
		original_accuracy = (self.total_attacks - self.skipped_attacks) * 100.0 / (self.total_attacks)
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
				self.successful_attacks * 100.0 / (self.successful_attacks + self.failed_attacks)
			)
		attack_success_rate = round(attack_success_rate, 2)
		return attack_success_rate