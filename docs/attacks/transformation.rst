==========================
Transformation
==========================

A transformation is a method of perturbing a text input by replacing words or phrases with ones similar in meaning. All transformations take a ``TokenizedText`` as input and return a list of type ``TokenizedText`` that contains possible transformations. Every transformation is a subclass of the abstract ``transformation`` class.

.. automodule:: textattack.transformations.transformation
   :members:


Composite Transformation 
--------------------------
Multiple transformations can be used by providing a list of ``transformation`` s to ``CompositeTransformation``

.. automodule:: textattack.transformations.composite_transformation
   :members:

Word Swap
-----------------
Word swap transformations act by replacing some words in the input. Subclasses can implement the abstract ``word_swap`` class by overriding ``self._get_replacement_words``

.. automodule:: textattack.transformations.word_swap
   :members:
   
   
Word Swap by Embedding
----------------------

.. automodule:: textattack.transformations.word_swap_embedding
   :members:
   
Word Swap by WordNet Word Replacement
---------------------------------------

.. automodule:: textattack.transformations.word_swap_wordnet
   :members:
   
Word Swap by Gradient
---------------------------------------

.. automodule:: textattack.transformations.word_swap_gradient_based
   :members:
   
Word Swap by Homoglyph
----------------------

.. automodule:: textattack.transformations.word_swap_homoglyph
   :members:
   
Word Swap by Neighboring Character Swap
---------------------------------------

.. automodule:: textattack.transformations.word_swap_neighboring_character_swap
   :members:
   
Word Swap by Random Character Deletion
---------------------------------------

.. automodule:: textattack.transformations.word_swap_random_character_deletion
   :members:
   
Word Swap by Random Character Insertion
---------------------------------------

.. automodule:: textattack.transformations.word_swap_random_character_insertion
   :members:
   
Word Swap by Random Character Substitution
---------------------------------------

.. automodule:: textattack.transformations.word_swap_random_character_substitution
   :members: