Augmenter Recipes API 
=====================

Summary: Transformations and constraints can be used for simple NLP data augmentations. 

In addition to the command-line interface, you can augment text dynamically by importing the
`Augmenter` in your own code. All `Augmenter` objects implement `augment` and `augment_many` to generate augmentations
of a string or a list of strings. Here's an example of how to use the `EmbeddingAugmenter` in a python script:

.. code-block:: python

   >>> from textattack.augmentation import EmbeddingAugmenter
   >>> augmenter = EmbeddingAugmenter()
   >>> s = 'What I cannot create, I do not understand.'
   >>> augmenter.augment(s)
   ['What I notable create, I do not understand.', 'What I significant create, I do not understand.', 'What I cannot engender, I do not understand.', 'What I cannot creating, I do not understand.', 'What I cannot creations, I do not understand.', 'What I cannot create, I do not comprehend.', 'What I cannot create, I do not fathom.', 'What I cannot create, I do not understanding.', 'What I cannot create, I do not understands.', 'What I cannot create, I do not understood.', 'What I cannot create, I do not realise.']

You can also create your own augmenter from scratch by importing transformations/constraints from `textattack.transformations` and `textattack.constraints`. Here's an example that generates augmentations of a string using `WordSwapRandomCharacterDeletion`:

.. code-block:: python

   >>> from textattack.transformations import WordSwapRandomCharacterDeletion
   >>> from textattack.transformations import CompositeTransformation
   >>> from textattack.augmentation import Augmenter
   >>> transformation = CompositeTransformation([WordSwapRandomCharacterDeletion()])
   >>> augmenter = Augmenter(transformation=transformation, transformations_per_example=5)
   >>> s = 'What I cannot create, I do not understand.'
   >>> augmenter.augment(s)
   ['What I cannot creae, I do not understand.', 'What I cannot creat, I do not understand.', 'What I cannot create, I do not nderstand.', 'What I cannot create, I do nt understand.', 'Wht I cannot create, I do not understand.']



Here is a list of recipes for NLP data augmentations

.. automodule:: textattack.augmentation.recipes
   :members:
   :noindex:
