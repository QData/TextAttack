Search Methods API Reference
============================

:class:`~textattack.search_methods.SearchMethod` attempts to find the optimal set of perturbations that will produce an adversarial example.
Finding such optimal perturbations becomes a combinatorial optimization problem, and search methods are typically heuristic search algorithms designed
to solve the underlying combinatorial problem.

More in-depth study of search algorithms for NLP adversarial attacks can be found in the following work 
`Searching for a Search Method: Benchmarking Search Algorithms for Generating NLP Adversarial Examples <https://arxiv.org/abs/2009.06368>`_
by Jin Yong Yoo, John X. Morris, Eli Lifland, and Yanjun Qi.

SearchMethod
------------
.. autoclass:: textattack.search_methods.SearchMethod
   :members:

BeamSearch
------------
.. autoclass:: textattack.search_methods.BeamSearch
   :members:

GreedySearch
------------
.. autoclass:: textattack.search_methods.GreedySearch
   :members:

GreedyWordSwapWIR
------------------
.. autoclass:: textattack.search_methods.GreedyWordSwapWIR
   :members:

AlzantotGeneticAlgorithm
-------------------------
.. autoclass:: textattack.search_methods.AlzantotGeneticAlgorithm
   :members:

ImprovedGeneticAlgorithm
-------------------------
.. autoclass:: textattack.search_methods.ImprovedGeneticAlgorithm
   :members:

ParticleSwarmOptimization
--------------------------
.. autoclass:: textattack.search_methods.ParticleSwarmOptimization
   :members:

