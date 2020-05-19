=====================
Search Methods
=====================

Search methods explore the transformation space in an attempt to find a successful attack as determined by a goal_function (:ref:`goal_function`) and possible constraints (:ref:`constraints`). All search methods are implemented as a subclass of the ``Attack`` class by implementing the ``attack_one`` method. 

Search Methods:

- `Greedy Search`_
- `Beam Search`_ 
- `Greedy Word Swap with Word Importance Ranking`_
- `Genetic Algorithm Word Swap`_


.. automodule:: textattack.search_methods.search_method
   :members:

Greedy Search
####################

.. automodule:: textattack.search_methods.greedy_word_swap
   :members:

Beam Search
############

.. automodule:: textattack.search_methods.beam_search 
   :members:


Greedy Word Swap with Word Importance Ranking
##############################################

.. automodule:: textattack.search_methods.greedy_word_swap_wir
   :members:
   
Genetic Algorithm Word Swap
###########################

.. automodule:: textattack.search_methods.genetic_algorithm
   :members:

