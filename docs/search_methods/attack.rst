=====================
Search Methods
=====================

Search methods explore the transformation space in an attempt to find a successful attack as determined by a goal_function (:ref:`goal_function`) and possible constraints (:ref:`constraints`). All search methods are implemented as a subclass of the ``Attack`` class by implementing the ``attack_one`` method. 

Search Methods:

- `Greedy Word Search`_
- `Beam Search`_ 
- `Genetic Algorithm`_

Attack
######################

.. automodule:: textattack.search_methods.attack
   :members:

Greedy Word Search
####################

.. automodule:: textattack.search_methods.greedy_word_swap
   :members:

.. automodule:: textattack.search_methods.greedy_word_swap_wir
   :members:

Beam Search
############

.. automodule:: textattack.search_methods.beam_search 
   :members:

Genetic Algorithm 
##################

.. automodule:: textattack.search_methods.genetic_algorithm
   :members:

