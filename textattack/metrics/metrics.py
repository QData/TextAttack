from abc import ABC, abstractmethod

import numpy as np
import torch

import textattack
from textattack.attack_results import (
    FailedAttackResult,
    MaximizedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)


class MetricDataFormat:
    """Metric data formats for consistent formatting across metrics."""

    STR = 0
    INT = 1
    FLOAT = 2
    PERCENTAGE = 3


class Metric(ABC):
    """A metric processes a list of `AttackResult` objects for a given
    statistic."""

    @property
    def key(self):
        """The string representation of the metric, like 'Attack Success Rate'
        or 'Levenshtein Distance'."""
        raise NotImplementedError()

    @property
    def data_format(self):
        """The metric format, which dictates how it is parsed to a string."""

        raise NotImplementedError()

    @staticmethod
    def compute(attack_results):
        """Returns a string representation of a metric calculated across a list
        of attack results."""
        raise NotImplementedError()

    @classmethod
    def compute_str(cls, attack_results):
        metric = cls.compute(attack_results)
        if cls.data_format == MetricDataFormat.STR:
            return metric
        elif cls.data_format == MetricDataFormat.INT:
            return str(metric)
        elif cls.data_format == MetricDataFormat.FLOAT:
            return str(round(metric, 4))
        elif cls.data_format == MetricDataFormat.PERCENTAGE:
            perc_str = str(round(metric * 100.0, 2))
            return f"{perc_str}%"
        else:
            raise ValueError(f"Invalid metric format {cls.data_format}")


class TotalAttacks(Metric):
    key = "Number of attacks"
    data_format = MetricDataFormat.INT

    @staticmethod
    def compute(attack_results):
        total = sum((1 for r in attack_results))
        return total


class TotalSuccessfulAttacks(Metric):
    key = "Successful attacks"
    data_format = MetricDataFormat.INT

    @staticmethod
    def compute(attack_results):
        return sum(
            (
                (
                    isinstance(r, SuccessfulAttackResult)
                    or isinstance(r, MaximizedAttackResult)
                )
                for r in attack_results
            )
        )


class TotalFailedAttacks(Metric):
    key = "Failed attacks"
    data_format = MetricDataFormat.INT

    @staticmethod
    def compute(attack_results):
        total = sum((isinstance(r, FailedAttackResult) for r in attack_results))
        return total


class TotalSkippedAttacks(Metric):
    key = "Skipped attacks"
    data_format = MetricDataFormat.INT

    @staticmethod
    def compute(attack_results):
        total = sum((isinstance(r, SkippedAttackResult) for r in attack_results))
        return total


class ModelAccuracy(Metric):
    key = "Model accuracy"
    data_format = MetricDataFormat.PERCENTAGE

    @staticmethod
    def compute(attack_results):
        total_attacks = len(attack_results)
        skipped_attacks = TotalSkippedAttacks.compute(attack_results)
        return (total_attacks - skipped_attacks) / (total_attacks)


class AccuracyUnderAttack(Metric):
    key = "Accuracy under attack"
    data_format = MetricDataFormat.PERCENTAGE

    @staticmethod
    def compute(attack_results):
        total_attacks = len(attack_results)
        failed_attacks = TotalFailedAttacks.compute(attack_results)
        return (failed_attacks) / (total_attacks)


class AttackSuccessRate(Metric):
    key = "Attack success rate"
    data_format = MetricDataFormat.PERCENTAGE

    @staticmethod
    def compute(attack_results):
        successful_attacks = TotalSuccessfulAttacks.compute(attack_results)
        failed_attacks = TotalFailedAttacks.compute(attack_results)
        if successful_attacks + failed_attacks == 0:
            return 0.0
        else:
            return successful_attacks / (successful_attacks + failed_attacks)


class AveragePerturbedWordPercentage(Metric):
    key = "Average perturbed word %"
    data_format = MetricDataFormat.PERCENTAGE

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
                perturbed_word_percentage = num_words_changed / len(
                    result.original_result.attacked_text.words
                )
            else:
                perturbed_word_percentage = 0
            perturbed_word_percentages.append(perturbed_word_percentage)

        perturbed_word_percentages = np.array(perturbed_word_percentages)

        return perturbed_word_percentages.mean()


class AverageNumberOfWords(Metric):
    key = "Average num. words per input"
    data_format = MetricDataFormat.FLOAT

    @staticmethod
    def compute(attack_results):
        num_words = np.array(
            [len(r.original_result.attacked_text.words) for r in attack_results]
        )
        return num_words.mean()


class AverageNumberOfQueries(Metric):
    key = "Average num. queries"
    data_format = MetricDataFormat.FLOAT

    @staticmethod
    def compute(attack_results):
        num_queries = np.array(
            [
                r.num_queries
                for r in attack_results
                if not isinstance(r, SkippedAttackResult)
            ]
        )
        return num_queries.mean()


class AverageUniversalSentenceEncoderCosSim(Metric):
    key = "Average USE cosine similarity"
    data_format = MetricDataFormat.FLOAT

    @staticmethod
    def compute(attack_results):
        use_model = (
            textattack.constraints.semantics.sentence_encoders.UniversalSentenceEncoder().model
        )
        original_texts = []
        perturbed_texts = []
        for r in attack_results:
            if isinstance(r, SuccessfulAttackResult):
                original_texts.append(r.original_result.attacked_text.text)
                perturbed_texts.append(r.perturbed_result.attacked_text.text)
        original_encodings = use_model(original_texts).numpy()
        original_encodings = torch.tensor(original_encodings)
        perturbed_encodings = use_model(perturbed_texts).numpy()
        perturbed_encodings = torch.tensor(perturbed_encodings)
        cos_sim_func = torch.nn.CosineSimilarity(dim=1)
        return cos_sim_func(original_encodings, perturbed_encodings).mean().item()


class AverageBLEUScore(Metric):
    key = "Average BLEU"
    data_format = MetricDataFormat.FLOAT

    @staticmethod
    def compute(attack_results):
        bleu_scorer = textattack.constraints.overlap.BLEU(0.0)
        scores = []
        for r in attack_results:
            if isinstance(r, SuccessfulAttackResult):
                text1 = r.original_result.attacked_text
                text2 = r.perturbed_result.attacked_text
                scores.append(bleu_scorer._score(text1, text2))
        return torch.tensor(scores).mean().item()
