"""

Stopword Modification
--------------------------

"""

import nltk

from textattack.constraints import PreTransformationConstraint
from textattack.shared.validators import transformation_consists_of_word_swaps
from textattack.shared import AttackedText
from textattack.transformations import Transformation
from typing import List, Set


class StopwordModification(PreTransformationConstraint):
    """A constraint disallowing the modification of stopwords."""

    def __init__(self, stopwords=None : Set[str]):
        if stopwords is not None:
            self.stopwords = set(stopwords)
        else:
            self.stopwords = set(nltk.corpus.stopwords.words("english"))

    def _get_modifiable_indices(self, current_text : AttackedText) -> Set[int]:
        """Returns the word indices in ``current_text`` which are able to be
        modified."""
        non_stopword_indices = set()
        for i, word in enumerate(current_text.words):
            if word not in self.stopwords:
                non_stopword_indices.add(i)
        return non_stopword_indices

    def check_compatibility(self, transformation : Transformation) -> bool:
        """The stopword constraint only is concerned with word swaps since
        paraphrasing phrases containing stopwords is OK.

        Args:
            transformation: The ``Transformation`` to check compatibility with.
        """
        return transformation_consists_of_word_swaps(transformation)
