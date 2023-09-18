"""
Metric Class
========================

"""

from abc import ABC, abstractmethod


class Metric(ABC):
    """A metric for evaluating results and data quality."""

    @abstractmethod
    def __init__(self, **kwargs):
        """Creates pre-built :class:`~textattack.Metric` that correspond to
        evaluation metrics for adversarial examples."""
        raise NotImplementedError()

    @abstractmethod
    def calculate(self, results):
        """Abstract function for computing any values which are to be calculated as a whole during initialization
        Args:
            results (``AttackResult`` objects):
                    Attack results for each instance in dataset
        """

        raise NotImplementedError
