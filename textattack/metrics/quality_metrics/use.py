from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.metrics import Metric


class USEMetric(Metric):
    """Calculates average USE similarity on all successfull attacks

    Args:
    results (:obj::`list`:class:`~textattack.goal_function_results.GoalFunctionResult`):
                    Attack results for each instance in dataset
    """

    def __init__(self, **kwargs):
        self.use_obj = UniversalSentenceEncoder()
        self.use_obj.model = UniversalSentenceEncoder()
        self.original_candidates = []
        self.successful_candidates = []
        self.all_metrics = {}

    def calculate(self, results):
        self.results = results

        for i, result in enumerate(self.results):
            if isinstance(result, FailedAttackResult):
                continue
            elif isinstance(result, SkippedAttackResult):
                continue
            else:
                self.original_candidates.append(result.original_result.attacked_text)
                self.successful_candidates.append(result.perturbed_result.attacked_text)

        use_scores = []
        for c in range(len(self.original_candidates)):
            use_scores.append(
                self.use_obj._sim_score(
                    self.original_candidates[c], self.successful_candidates[c]
                ).item()
            )

        self.all_metrics["avg_attack_use_score"] = round(
            sum(use_scores) / len(use_scores), 2
        )

        return self.all_metrics
