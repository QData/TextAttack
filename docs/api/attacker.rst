Attacker API Reference
=======================

Attacker
-------------
While :class:`~textattack.Attack` is the main class used to carry out the adversarial attack, it is only useful for attacking one example at a time.
It lacks features that support attacking multiple samples in parallel (i.e. multi-GPU), saving checkpoints, or logging results to text file, CSV file, or wandb.
:class:`~textattack.Attacker` provides these features in an easy-to-use API.

.. autoclass:: textattack.Attacker
   :members:


AttackArgs
-------------
:class:`~textattack.AttackArgs` represents arguments to be passed to :class:`~textattack.Attacker`, such as number of examples to attack, interval at which to save checkpoints, logging details.

.. autoclass:: textattack.AttackArgs
   :members:
