"""
Attack Metrics Class
========================

"""

from abc import ABC, abstractmethod

from textattack.attack_results import AttackResult


class AttackMetric(AttackResult, ABC):
    """A metric for evaluating Adversarial Attack candidates."""

    @staticmethod
    @abstractmethod
    def __init__(self,results,**kwargs):
        """Creates pre-built :class:`~textattack.AttackMetric` that correspond to
        evaluation metrics for adversarial examples.
        """
        raise NotImplementedError()
        
    @staticmethod
    @abstractmethod
    def calculate():
        """ Abstract function for computing any values which are to be calculated as a whole during initialization"""
        raise NotImplementedError