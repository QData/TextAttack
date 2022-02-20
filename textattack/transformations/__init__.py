""".. _transformations:

Transformations
==========================

A transformation is a method which perturbs a text input through the insertion, deletion and substiution of words, characters, and phrases. All transformations take a ``TokenizedText`` as input and return a list of ``TokenizedText`` that contains possible transformations. Every transformation is a subclass of the abstract ``Transformation`` class.
"""
from .transformation import Transformation

from .sentence_transformations import *
from .word_swaps import *
from .word_insertions import *
from .word_merges import *

from .composite_transformation import CompositeTransformation
from .word_deletion import WordDeletion
from .word_innerswap_random import WordInnerSwapRandom
