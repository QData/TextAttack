"""
Attack Metric Quality Recipes:
==============================

"""

from textattack.metrics.quality_metrics.bert_score import BERTScoreMetric
from textattack.metrics.quality_metrics.meteor_score import MeteorMetric
from textattack.metrics.quality_metrics.perplexity import Perplexity
from textattack.metrics.quality_metrics.sentence_bert import SBERTMetric
from textattack.metrics.quality_metrics.use import USEMetric

from .metric import Metric


class AdvancedAttackMetric(Metric):
    """Calculate a suite of advanced metrics to evaluate attackResults'
    quality."""

    def __init__(self, choices: list[str] = ["use"]):
        self.achoices = choices
        available_metrics = {
            "use": USEMetric,
            "perplexity": Perplexity,
            "bert_score": BERTScoreMetric,
            "meteor_score": MeteorMetric,
            "sbert_score": SBERTMetric,
        }
        self.selected_metrics = {}
        for choice in self.achoices:
            if choice not in available_metrics:
                raise KeyError(f"'{choice}' is not a valid metric name")
            metric = available_metrics[choice]()
            self.selected_metrics.update({choice: metric})

    def add_metric(self, name: str, metric: Metric):
        if not isinstance(metric, Metric):
            raise ValueError(f"Object {metric} must be a subtype of Metric")
        self.selected_metrics.update({name: metric})

    def calculate(self, results) -> dict[str, float]:
        advanced_metrics = {}
        # TODO: Would like to guarantee unique keys from calls to calculate()
        for metric in self.selected_metrics.values():
            advanced_metrics.update(metric.calculate(results))
        return advanced_metrics
