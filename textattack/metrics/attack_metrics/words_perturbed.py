"""

Metrics on perturbed words
---------------------------------------------------------------------

"""

import numpy as np

from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.metrics import Metric


class WordsPerturbed(Metric):
    def __init__(self):
        self.total_attacks = 0
        self.all_num_words = None
        self.perturbed_word_percentages = None
        self.num_words_changed_until_success = 0
        self.all_metrics = {}

    def calculate(self, results):
        """Calculates all metrics related to perturbed words in an attack.

        Args:
            results (``AttackResult`` objects):
                Attack results for each instance in dataset
        """

        self.results = results
        self.total_attacks = len(self.results)
        self.all_num_words = np.zeros(len(self.results))
        self.perturbed_word_percentages = np.zeros(len(self.results))
        self.num_words_changed_until_success = np.zeros(2**16)
        self.max_words_changed = 0

        for i, result in enumerate(self.results):
            self.all_num_words[i] = len(result.original_result.attacked_text.words)

            if isinstance(result, FailedAttackResult) or isinstance(
                result, SkippedAttackResult
            ):
                continue

            num_words_changed = len(
                result.original_result.attacked_text.all_words_diff(
                    result.perturbed_result.attacked_text
                )
            )
            self.num_words_changed_until_success[num_words_changed - 1] += 1
            self.max_words_changed = max(
                self.max_words_changed or num_words_changed, num_words_changed
            )
            if len(result.original_result.attacked_text.words) > 0:
                perturbed_word_percentage = (
                    num_words_changed
                    * 100.0
                    / len(result.original_result.attacked_text.words)
                )
            else:
                perturbed_word_percentage = 0

            self.perturbed_word_percentages[i] = perturbed_word_percentage

        self.all_metrics["avg_word_perturbed"] = self.avg_number_word_perturbed_num()
        self.all_metrics["avg_word_perturbed_perc"] = self.avg_perturbation_perc()
        self.all_metrics["max_words_changed"] = self.max_words_changed
        self.all_metrics[
            "num_words_changed_until_success"
        ] = self.num_words_changed_until_success

        return self.all_metrics

    def avg_number_word_perturbed_num(self):
        average_num_words = self.all_num_words.mean()
        average_num_words = round(average_num_words, 2)
        return average_num_words

    def avg_perturbation_perc(self):
        self.perturbed_word_percentages = self.perturbed_word_percentages[
            self.perturbed_word_percentages > 0
        ]
        average_perc_words_perturbed = self.perturbed_word_percentages.mean()
        average_perc_words_perturbed = round(average_perc_words_perturbed, 2)
        return average_perc_words_perturbed
