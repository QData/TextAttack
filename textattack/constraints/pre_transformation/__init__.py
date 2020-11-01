""".. _pre_transformation:

Pre-Transformation:
---------------------

Pre-transformation constraints determine if a transformation is valid based on only the original input and the position of the replacement. These constraints are applied before the transformation is even called. For example, these constraints can prevent search methods from swapping words at the same index twice, or from replacing stopwords.
"""
from .stopword_modification import StopwordModification
from .repeat_modification import RepeatModification
from .input_column_modification import InputColumnModification
from .max_word_index_modification import MaxWordIndexModification
from .min_word_length import MinWordLength
