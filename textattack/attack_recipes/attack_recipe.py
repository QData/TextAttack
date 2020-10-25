"""
We provide a number of pre-built attack recipes, which correspond to attacks from the literature. To run an attack recipe from the command line, run::

    textattack attack --recipe [recipe_name]

To initialize an attack in Python script, use::

    <recipe name>.build(model_wrapper)

For example, ``attack = InputReductionFeng2018.build(model)`` creates `attack`, an object of type ``Attack`` with the goal function, transformation, constraints, and search method specified in that paper. This object can then be used just like any other attack; for example, by calling ``attack.attack_dataset``.

TextAttack supports the following attack recipes (each recipe's documentation contains a link to the corresponding paper): 

.. contents:: :local:
"""

from abc import ABC, abstractmethod

from textattack.shared import Attack


class AttackRecipe(Attack, ABC):
    """A recipe for building an NLP adversarial attack from the literature."""

    @staticmethod
    @abstractmethod
    def build(model):
        """Creates an attack recipe from recipe-specific arguments.

        Allows for support of different configurations of a single
        attack.
        """
        raise NotImplementedError()
