"""
Attack Metric Quality Recipes:
==============================

"""
import random

from . import metric

class AdvancedAttackMetric(results):
    """Calculate a suite of advanced metrics to evaluate attackResults' quality 
    """

    def __init__(self, results, **kwargs):
        perplexity_stats = Perplexity().calculate(results)
        use_stats = USEMetric().calculate(results)
        bert_score = BERTScoreMetric().calculate(results)
        meteor_score = MeteorMetric().calculate(results)
        sbert_score = SBERTMetric().calculate(results)
        return perplexity_stats, use_stats, bert_score, meteor_score, sbert_score