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

    def __init__(self, choices=["use"]):
        self.achoices = choices

    def calculate(self, results):
        advanced_metrics = {}
        if "use" in self.achoices:
            advanced_metrics.update(USEMetric().calculate(results))
        if "perplexity" in self.achoices:
            advanced_metrics.update(Perplexity().calculate(results))
        if "bert_score" in self.achoices:
            advanced_metrics.update(BERTScoreMetric().calculate(results))
        if "meteor_score" in self.achoices:
            advanced_metrics.update(MeteorMetric().calculate(results))
        if "sbert_score" in self.achoices:
            advanced_metrics.update(SBERTMetric().calculate(results))
        return advanced_metrics
