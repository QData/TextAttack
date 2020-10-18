==========================
Transformations
==========================

A transformation is a method which perturbs a text input through the insertion, deletion and substiution of words, characters, and phrases. All transformations take a ``TokenizedText`` as input and return a list of ``TokenizedText``\s that contains possible transformations. Every transformation is a subclass of the abstract ``Transformation`` class.

.. automodule:: textattack.transformations.transformation
   :special-members: __call__
   :private-members:
   :members:


Composite Transformation 
--------------------------
Multiple transformations can be used by providing a list of ``Transformation``\s to ``CompositeTransformation``

.. automodule:: textattack.transformations.composite_transformation
   :members:

Word Swap
-----------------
Word swap transformations act by replacing some words in the input. Subclasses can implement the abstract ``WordSwap`` class by overriding ``self._get_replacement_words``

.. automodule:: textattack.transformations.word_swap
   :private-members:
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

.. automodule:: textattack.transformations.word_swap_homoglyph_swap
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
-------------------------------------------

.. automodule:: textattack.transformations.word_swap_random_character_substitution
   :members:

Word Swap by Changing Location
-------------------------------

.. automodule:: textattack.transformations.word_swap_change_location
   :members:
   
Word Swap by Changing Number
-------------------------------

.. automodule:: textattack.transformations.word_swap_change_number
   :members:
   
Word Swap by Changing Name
-------------------------------

.. automodule:: textattack.transformations.word_swap_change_name
   :members:
   
Word Swap by Contraction
-------------------------------

.. automodule:: textattack.transformations.word_swap_contract
   :members:
   
Word Swap by Extension
-------------------------------

.. automodule:: textattack.transformations.word_swap_extend
   :members:
