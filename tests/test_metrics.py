from collections import OrderedDict

import numpy as np
import pytest

import textattack
from textattack import metrics
from textattack.goal_function_results import GoalFunctionResultStatus

data = OrderedDict(
    [
        ("raw_output", np.array([0.08, 0.92])),
        ("output", 1),
        ("score", 0.98),
        ("num_queries", 30),
        ("goal_status", GoalFunctionResultStatus.SUCCEEDED),
        ("ground_truth_output", 0),
    ]
)


def attacked_text(s):
    return textattack.shared.AttackedText(s)


@pytest.fixture
def attack_results():
    original_attacked_text = attacked_text("effective but too-tepid biopic")
    original_text = textattack.goal_function_results.ClassificationGoalFunctionResult(
        *([original_attacked_text] + list(data.values()))
    )
    perturbed_attacked_text = attacked_text("compelling but too-tepid film")
    perturbed_text = textattack.goal_function_results.ClassificationGoalFunctionResult(
        *([perturbed_attacked_text] + list(data.values()))
    )
    ar1 = textattack.attack_results.SuccessfulAttackResult(
        original_text, perturbed_text
    )
    ar2 = textattack.attack_results.SuccessfulAttackResult(
        original_text, perturbed_text
    )
    ar2.num_queries = 50
    ar3 = textattack.attack_results.SuccessfulAttackResult(
        original_text, perturbed_text
    )
    ar3.num_queries = 70
    ar4 = textattack.attack_results.FailedAttackResult(original_text, perturbed_text)
    ar5 = textattack.attack_results.FailedAttackResult(original_text, perturbed_text)
    ar6 = textattack.attack_results.SkippedAttackResult(original_text)
    return [ar1, ar2, ar3, ar4, ar5, ar6]


class TestMetrics:
    def test_total_attacks(self, attack_results):
        metric = metrics.TotalAttacks
        assert metric.key == "Number of attacks"
        assert metric.compute_str(attack_results) == "6"

    def test_successful_attacks(self, attack_results):
        metric = metrics.TotalSuccessfulAttacks
        assert metric.key == "Successful attacks"
        assert metric.compute_str(attack_results) == "3"

    def test_failed_attacks(self, attack_results):
        metric = metrics.TotalFailedAttacks
        assert metric.key == "Failed attacks"
        assert metric.compute_str(attack_results) == "2"

    def test_skipped_attacks(self, attack_results):
        metric = metrics.TotalSkippedAttacks
        assert metric.key == "Skipped attacks"
        assert metric.compute_str(attack_results) == "1"

    def test_model_accuracy(self, attack_results):
        metric = metrics.ModelAccuracy
        assert metric.key == "Model accuracy"
        assert metric.compute_str(attack_results) == "83.33%"

    def test_accuracy_under_attack(self, attack_results):
        metric = metrics.AccuracyUnderAttack
        assert metric.key == "Accuracy under attack"
        assert metric.compute_str(attack_results) == "33.33%"

    def test_attack_success_rate(self, attack_results):
        metric = metrics.AttackSuccessRate
        assert metric.key == "Attack success rate"
        assert metric.compute_str(attack_results) == "60.0%"

    def test_average_perturbed_word_percentage(self, attack_results):
        metric = metrics.AveragePerturbedWordPercentage
        assert metric.key == "Average perturbed word %"
        assert metric.compute_str(attack_results) == "50.0%"

    def test_average_num_words(self, attack_results):
        metric = metrics.AverageNumberOfWords
        assert metric.key == "Average num. words per input"
        assert metric.compute_str(attack_results) == "4.0"

    def test_average_num_queries(self, attack_results):
        metric = metrics.AverageNumberOfQueries
        assert metric.key == "Average num. queries"
        assert metric.compute_str(attack_results) == "24.0"

    def test_average_use(self, attack_results):
        metric = metrics.AverageUniversalSentenceEncoderCosSim
        assert metric.key == "Average USE cosine similarity"
        assert metric.compute_str(attack_results) == "0.5029"

    def test_average_bleu(self, attack_results):
        metric = metrics.AverageBLEUScore
        assert metric.key == "Average BLEU"
        assert metric.compute_str(attack_results) == "0.0"
