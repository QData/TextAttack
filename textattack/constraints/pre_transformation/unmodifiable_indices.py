from textattack.constraints import PreTransformationConstraint


class UnmodifiableIndices(PreTransformationConstraint):
    """A constraint that prevents the modification of certain words at specific
    indices.

    Args:
        indices (list(int)): A list of indices which are unmodifiable
    """

    def __init__(self, indices):
        self.unmodifiable_indices = indices

    def _get_modifiable_indices(self, current_text):
        unmodifiable_set = current_text.convert_from_original_idxs(
            self.unmodifiable_indices
        )
        return set(
            i for i in range(0, len(current_text.words)) if i not in unmodifiable_set
        )

    def extra_repr_keys(self):
        return ["indices"]
