Attack API Reference
=======================

Attack
------------
Attack is composed of four components:

- `Goal Functions <../attacks/goal_function.html>`__ stipulate the goal of the attack, like to change the prediction score of a classification model, or to change all of the words in a translation output.
- `Constraints <../attacks/constraint.html>`__ determine if a potential perturbation is valid with respect to the original input.
- `Transformations <../attacks/transformation.html>`__ take a text input and transform it by inserting and deleting characters, words, and/or phrases.
- `Search Methods <../attacks/search_method.html>`__ explore the space of possible **transformations** within the defined **constraints** and attempt to find a successful perturbation which satisfies the **goal function**.

The :class:`~textattack.Attack` class represents an adversarial attack composed of a goal function, search method, transformation, and constraints.

.. autoclass:: textattack.Attack
   :members:

AttackRecipe
-------------
Attack recipe is a subclass of :class:`~textattack.Attack` class that has a special method :meth:`build` which 
returns a pre-built :class:`~textattack.Attack` that correspond to attacks from the literature.

 

.. autoclass:: textattack.attack_recipes.AttackRecipe
   :members:

Attacker
-------------
:class:`~textattack.Attack` class is only useful for attacking one example at a time. 
However, when attacking a dataset, we typically want other useful features such as parallel processing, saving/resuming from a checkpint, logging to files.
:class:`~textattack.Attacker` provides these features in an easy-to-use API. 

.. autoclass:: textattack.Attacker
   :members:


AttackArgs
-------------
Attack arguments to be passed to :class:`~textattack.Attacker`.

.. autoclass:: textattack.AttackArgs
   :members:
