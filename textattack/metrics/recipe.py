"""
Attack Metric Quality Recipes:
==============================

"""

from __future__ import annotations

from textattack.metrics.quality_metrics.bert_score import BERTScoreMetric
from textattack.metrics.quality_metrics.meteor_score import MeteorMetric
from textattack.metrics.quality_metrics.perplexity import Perplexity
from textattack.metrics.quality_metrics.sentence_bert import SBERTMetric
from textattack.metrics.quality_metrics.use import USEMetric

from .metric import Metric


class AdvancedAttackMetric(Metric):
    """Calculate a suite of advanced metrics to evaluate attackResults'
    quality."""

    _AVAILABLE_METRICS = {
        "use": USEMetric,
        "perplexity": Perplexity,
        "bert_score": BERTScoreMetric,
        "meteor_score": MeteorMetric,
        "sbert_score": SBERTMetric,
    }

    def __init__(self, choices: list[str] = ["use"]):
        self.selected_metrics = {}
        for choice in choices:
            if choice not in self._AVAILABLE_METRICS:
                raise ValueError(f"'{choice}' is not a valid metric name")
            # Construction (and any pretrained model loading) is deferred
            # until `calculate()` actually needs the metric.
            self.selected_metrics[choice] = self._AVAILABLE_METRICS[choice]

    @property
    def achoices(self):
        return list(self.selected_metrics.keys())

    def add_metric(self, name: str, metric: Metric):
        if not isinstance(metric, Metric):
            raise ValueError(f"Object {metric} must be a subtype of Metric")
        self.selected_metrics[name] = metric

    def calculate(self, results) -> dict[str, float]:
        advanced_metrics = {}
        # TODO: Would like to guarantee unique keys from calls to calculate()
        for name, metric in self.selected_metrics.items():
            if isinstance(metric, type):
                metric = metric()
                self.selected_metrics[name] = metric
            advanced_metrics.update(metric.calculate(results))
        return advanced_metrics
