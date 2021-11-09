""".. _augmentation:

TextAttack augmentation package:
=================================

Transformations and constraints can be used outside of an attack for simple NLP data augmentation with the ``Augmenter`` class that returns all possible transformations for a given string.
"""


from .augmenter import Augmenter
from .recipes import (
    WordNetAugmenter,
    EmbeddingAugmenter,
    CharSwapAugmenter,
    EasyDataAugmenter,
    CheckListAugmenter,
    DeletionAugmenter,
    CLAREAugmenter,
    BackTranslationAugmenter,
)
