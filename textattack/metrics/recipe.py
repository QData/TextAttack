"""
Attack Metric Quality Recipes:
==============================

"""
import random

from . import metric

class AdvancedAttackMetric(Metric):
    """Calculate a suite of advanced metrics to evaluate attackResults' quality 
    """

    def __init__(self, choices=['use']):
        self.achoices = choices

    def calculate(self, results):
        advanced_metrics = {}
        if 'use' in self.achoices: 
            advanced_metrics['use_stats'] = USEMetric().calculate(results)
        if 'perplexity' in self.achoices: 
            advanced_metrics['perplexity_stats'] = Perplexity().calculate(results)
        if 'bert_score' in self.achoices: 
            advanced_metrics['bert_score'] = BERTScoreMetric().calculate(results)
        if 'meteor_score' in self.achoices: 
            advanced_metrics['meteor_score'] = MeteorMetric().calculate(results)
        if 'sbert_score' in self.achoices: 
            advanced_metrics['sbert_score'] = SBERTMetric().calculate(results)
        return advanced_metrics