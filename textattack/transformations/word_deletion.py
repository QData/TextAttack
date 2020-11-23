"""
word deletion Transformation
============================================

"""

from typing import List, Set

from textattack.shared import AttackedText

from .transformation import Transformation


class WordDeletion(Transformation):
    """An abstract class that takes a sentence and transforms it by deleting a
    single word.

    letters_to_insert (string): letters allowed for insertion into words
    """

    def _get_transformations(
        self, current_text: AttackedText, indices_to_modify: Set[int]
    ) -> List[AttackedText]:
        # words = current_text.words
        transformed_texts = []
        if len(current_text.words) > 1:
            for i in indices_to_modify:
                transformed_texts.append(current_text.delete_word_at_index(i))
        return transformed_texts
