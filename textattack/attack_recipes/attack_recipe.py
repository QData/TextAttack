"""
Attack Recipe Class
========================

"""

from abc import ABC, abstractmethod

from textattack import Attack


class AttackRecipe(Attack, ABC):
    """A recipe for building an NLP adversarial attack from the literature."""

    @staticmethod
    @abstractmethod
    def build(model_wrapper):
        """Creates an attack recipe from recipe-specific arguments.

        Allows for support of different configurations of a single
        attack.
        """
        raise NotImplementedError()
