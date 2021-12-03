Attack API Reference
=======================

Attack
------------
Attack is composed of four components:

- :ref:`Goal Functions  <goal_functions>`: stipulate the goal of the attack, like to change the prediction score of a classification model, or to change all of the words in a translation output.
- :ref:`Constraints <constraint>`: determine if a potential perturbation is valid with respect to the original input.
- :ref:`Transformations  <transformations>`: take a text input and transform it by inserting and deleting characters, words, and/or phrases.
- :ref:`Search Methods  <search_methods>`: explore the space of possible **transformations** within the defined **constraints** and attempt to find a successful perturbation which satisfies the **goal function**.

The :class:`~textattack.Attack` class represents an adversarial attack composed of a goal function, search method, transformation, and constraints.

.. autoclass:: textattack.Attack
   :members:

AttackRecipe
-------------
Attack recipe is a subclass of :class:`~textattack.Attack` class that has a special method :meth:`build` which 
returns a pre-built :class:`~textattack.Attack` that correspond to attacks from the literature.

 

.. autoclass:: textattack.attack_recipes.AttackRecipe
   :members:
