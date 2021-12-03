"""
WordInsertion Class
-------------------------------
Word Insertion transformations act by inserting a new word at a specific word index.
For example, if we insert "new" in position 3 in the text "I like the movie", we get "I like the new movie".
Subclasses can implement the abstract ``WordInsertion`` class by overriding ``self._get_new_words``.
"""
from textattack.transformations import Transformation


class WordInsertion(Transformation):
    """A base class for word insertions."""

    def _get_new_words(self, current_text, index):
        """Returns a set of new words we can insert at position `index` of `current_text`
        Args:
            current_text (AttackedText): Current text to modify.
            index (int): Position in which to insert a new word
        Returns:
            list[str]: List of new words to insert.
        """
        raise NotImplementedError()

    def _get_transformations(self, current_text, indices_to_modify):
        """
        Return a set of transformed texts obtained by insertion a new word in `indices_to_modify`
        Args:
            current_text (AttackedText): Current text to modify.
            indices_to_modify (list[int]): List of positions in which to insert a new word.

        Returns:
            list[AttackedText]: List of transformed texts
        """
        transformed_texts = []

        for i in indices_to_modify:
            new_words = self._get_new_words(current_text, i)

            new_transformted_texts = []
            for w in new_words:
                new_transformted_texts.append(
                    current_text.insert_text_before_word_index(i, w)
                )
            transformed_texts.extend(new_transformted_texts)

        return transformed_texts
