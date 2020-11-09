""".. _transformations:

Transformations
==========================

A transformation is a method which perturbs a text input through the insertion, deletion and substiution of words, characters, and phrases. All transformations take a ``TokenizedText`` as input and return a list of ``TokenizedText`` that contains possible transformations. Every transformation is a subclass of the abstract ``Transformation`` class.
"""

from .transformation import Transformation
from .composite_transformation import CompositeTransformation
from .word_swap import WordSwap

# Black-box transformations
from .word_deletion import WordDeletion
from .word_swap_embedding import WordSwapEmbedding
from .word_swap_hownet import WordSwapHowNet
from .word_swap_homoglyph_swap import WordSwapHomoglyphSwap
from .word_swap_inflections import WordSwapInflections
from .word_swap_neighboring_character_swap import WordSwapNeighboringCharacterSwap
from .word_swap_random_character_deletion import WordSwapRandomCharacterDeletion
from .word_swap_random_character_insertion import WordSwapRandomCharacterInsertion
from .word_swap_random_character_substitution import WordSwapRandomCharacterSubstitution
from .word_swap_wordnet import WordSwapWordNet
from .word_swap_masked_lm import WordSwapMaskedLM
from .word_swap_random_word import RandomSwap
from .random_synonym_insertion import RandomSynonymInsertion
from .word_swap_qwerty import WordSwapQWERTY
from .word_swap_contract import WordSwapContract
from .word_swap_extend import WordSwapExtend
from .word_swap_change_number import WordSwapChangeNumber
from .word_swap_change_location import WordSwapChangeLocation
from .word_swap_change_name import WordSwapChangeName

# White-box transformations
from .word_swap_gradient_based import WordSwapGradientBased
