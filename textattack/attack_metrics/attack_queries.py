import numpy as np

from textattack.attack_results import SkippedAttackResult

from .attack_metric import AttackMetric


class AttackQueries(AttackMetric):
    """Calculates all metrics related to number of queries in an attack

    Args:
    results (:obj::`list`:class:`~textattack.goal_function_results.GoalFunctionResult`):
                    Attack results for each instance in dataset
    """

    def __init__(self, results):
        self.results = results

        self.calculate()

    def calculate(self):
        self.num_queries = np.array(
            [
                r.num_queries
                for r in self.results
                if not isinstance(r, SkippedAttackResult)
            ]
        )

    def avg_num_queries_num(self):
        avg_num_queries = self.num_queries.mean()
        avg_num_queries = round(avg_num_queries, 2)
        return avg_num_queries
