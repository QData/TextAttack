"""
Word Merge
============================================
Word Merge transformations act by taking two adjacent words, and "merges" them into one word by deleting one word and replacing another.
For example, if we can merge the words "the" and "movie" in the text "I like the movie" and get following text: "I like film".
When we choose to "merge" word at index ``i``, we merge it with the next word at ``i+1``.
"""
from textattack.transformations import Transformation


class WordMerge(Transformation):
    """An abstract class for word merges."""

    def __call__(
        self,
        current_text,
        pre_transformation_constraints=[],
        indices_to_modify=None,
        shifted_idxs=True,
    ):
        """Returns a list of all possible transformations for ``current_text``.
        Applies the ``pre_transformation_constraints`` then calls
        ``_get_transformations``.

        Args:
            current_text: The ``AttackedText`` to transform.
            pre_transformation_constraints: The ``PreTransformationConstraint`` to apply before
                beginning the transformation.
            indices_to_modify: Which word indices should be modified as dictated by the
                ``SearchMethod``.
            shifted_idxs (bool): Whether indices have been shifted from
                their original position in the text.
        """
        if indices_to_modify is None:
            indices_to_modify = set(range(len(current_text.words) - 1))
        else:
            indices_to_modify = set(indices_to_modify)

        if shifted_idxs:
            indices_to_modify = set(
                current_text.convert_from_original_idxs(indices_to_modify)
            )

        for constraint in pre_transformation_constraints:
            allowed_indices = constraint(current_text, self)
            for i in indices_to_modify:
                if i not in allowed_indices and i + 1 not in allowed_indices:
                    indices_to_modify.remove(i)

        transformed_texts = self._get_transformations(current_text, indices_to_modify)
        for text in transformed_texts:
            text.attack_attrs["last_transformation"] = self
            if len(text.attack_attrs["newly_modified_indices"]) == 0:
                print("xcv", text, len(text.attack_attrs["newly_modified_indices"]))
        return transformed_texts

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

            for w in new_words:
                temp_text = current_text.replace_word_at_index(i, w)
                transformed_texts.append(temp_text.delete_word_at_index(i + 1))

        return transformed_texts
