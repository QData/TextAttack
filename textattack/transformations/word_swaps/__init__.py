"""
word_swaps package
-------------------------------

"""


from .word_swap import WordSwap

# Black box transformations
from .chn_transformations import *
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
from .word_swap_qwerty import WordSwapQWERTY
from .word_swap_contract import WordSwapContract
from .word_swap_extend import WordSwapExtend
from .word_swap_change_number import WordSwapChangeNumber
from .word_swap_change_location import WordSwapChangeLocation
from .word_swap_change_name import WordSwapChangeName

# White box transformation
from .word_swap_gradient_based import WordSwapGradientBased
