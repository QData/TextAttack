"""
Metric Class
========================

"""

from abc import ABC, abstractmethod


class Metric(ABC):
    """A metric for evaluating Adversarial Attack candidates."""

    @abstractmethod
    def __init__(self, results, **kwargs):
        """Creates pre-built :class:`~textattack.AttackMetric` that correspond to
        evaluation metrics for adversarial examples.
        """
        raise NotImplementedError()

    @abstractmethod
    def calculate():
        """ Abstract function for computing any values which are to be calculated as a whole during initialization"""
        raise NotImplementedError
