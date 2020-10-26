"""

Input Column Modification
--------------------------

"""

from textattack.constraints import PreTransformationConstraint


class InputColumnModification(PreTransformationConstraint):
    """A constraint disallowing the modification of words within a specific
    input column.

    For example, can prevent modification of 'premise' during
    entailment.
    """

    def __init__(self, matching_column_labels, columns_to_ignore):
        self.matching_column_labels = matching_column_labels
        self.columns_to_ignore = columns_to_ignore

    def _get_modifiable_indices(self, current_text):
        """Returns the word indices in current_text which are able to be
        deleted.

        If ``current_text.column_labels`` doesn't match
            ``self.matching_column_labels``, do nothing, and allow all words
            to be modified.

        If it does match, only allow words to be modified if they are not
            in columns from ``columns_to_ignore``.
        """
        if current_text.column_labels != self.matching_column_labels:
            return set(range(len(current_text.words)))

        idx = 0
        indices_to_modify = set()
        for column, words in zip(
            current_text.column_labels, current_text.words_per_input
        ):
            num_words = len(words)
            if column not in self.columns_to_ignore:
                indices_to_modify |= set(range(idx, idx + num_words))
            idx += num_words
        return indices_to_modify

    def extra_repr_keys(self):
        return ["matching_column_labels", "columns_to_ignore"]
