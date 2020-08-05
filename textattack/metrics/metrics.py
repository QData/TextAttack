from abc import ABC, abstractmethod

import numpy as np

from textattack.attack_results import (
    FailedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)


class Metric(ABC):
    """A metric processes a list of `AttackResult` objects for a given
    statistic."""

    @property
    def key(self):
        """The string representation of the metric, like 'Attack Success Rate'
        or 'Levenshtein Distance'."""
        raise NotImplementedError()

    @staticmethod
    def compute(attack_results):
        """Returns a string representation of a metric calculated across a list
        of attack results."""
        raise NotImplementedError()


class TotalAttacks(Metric):
    key = "Number of attacks"

    @staticmethod
    def compute(attack_results):
        total = sum((1 for r in attack_results))
        return str(total)


class TotalSuccessfulAttacks(Metric):
    key = "Successful attacks"

    @staticmethod
    def compute(attack_results):
        return sum((isinstance(r, SuccessfulAttackResult) for r in attack_results))


class TotalFailedAttacks(Metric):
    key = "Failed attacks"

    @staticmethod
    def compute(attack_results):
        total = sum((isinstance(r, FailedAttackResult) for r in attack_results))
        return total


class TotalSkippedAttacks(Metric):
    key = "Skipped attacks"

    @staticmethod
    def compute(attack_results):
        total = sum((isinstance(r, SkippedAttackResult) for r in attack_results))
        return total


class ModelAccuracy(Metric):
    key = "Model accuracy"

    @staticmethod
    def compute(attack_results):
        total_attacks = len(attack_results)
        skipped_attacks = TotalSkippedAttacks.compute(attack_results)
        model_accuracy = (total_attacks - skipped_attacks) * 100.0 / (total_attacks)
        return str(round(model_accuracy, 2)) + "%"


class AccuracyUnderAttack(Metric):
    key = "Accuracy under attack"

    @staticmethod
    def compute(attack_results):
        total_attacks = len(attack_results)
        failed_attacks = TotalFailedAttacks.compute(attack_results)
        accuracy_under_attack = (failed_attacks) * 100.0 / (total_attacks)
        return str(round(accuracy_under_attack, 2)) + "%"


class AttackSuccessRate(Metric):
    key = "Attack success rate"

    @staticmethod
    def compute(attack_results):
        successful_attacks = TotalSuccessfulAttacks.compute(attack_results)
        failed_attacks = TotalFailedAttacks.compute(attack_results)
        if successful_attacks + failed_attacks == 0:
            attack_success_rate = 0
        else:
            attack_success_rate = (
                successful_attacks * 100.0 / (successful_attacks + failed_attacks)
            )
        return str(round(attack_success_rate, 2)) + "%"


class AveragePerturbedWordPercentage(Metric):
    key = "Average perturbed word %"

    @staticmethod
    def compute(attack_results):
        perturbed_word_percentages = []
        for result in attack_results:

            if not isinstance(result, SuccessfulAttackResult):
                continue

            num_words_changed = len(
                result.original_result.attacked_text.all_words_diff(
                    result.perturbed_result.attacked_text
                )
            )

            if len(result.original_result.attacked_text.words) > 0:
                perturbed_word_percentage = (
                    num_words_changed
                    * 100.0
                    / len(result.original_result.attacked_text.words)
                )
            else:
                perturbed_word_percentage = 0
            perturbed_word_percentages.append(perturbed_word_percentage)

        perturbed_word_percentages = np.array(perturbed_word_percentages)
        print("perturbed_word_percentages ->", perturbed_word_percentages)

        average_perc_words_perturbed = perturbed_word_percentages.mean()
        return str(round(average_perc_words_perturbed, 2)) + "%"


class AverageNumberOfWords(Metric):
    key = "Average num. words per input"

    @staticmethod
    def compute(attack_results):
        num_words = np.array(
            [len(r.original_result.attacked_text.words) for r in attack_results]
        )
        avg_num_words = num_words.mean()
        return str(round(avg_num_words, 2))


class AverageNumberOfQueries(Metric):
    key = "Average num. queries"

    @staticmethod
    def compute(attack_results):
        num_queries = np.array(
            [
                r.num_queries
                for r in attack_results
                if not isinstance(r, SkippedAttackResult)
            ]
        )
        avg_num_queries = num_queries.mean()
        return str(round(avg_num_queries, 2))
