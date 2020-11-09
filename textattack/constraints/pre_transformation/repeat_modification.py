"""
Repeat Modification
--------------------------

"""

from textattack.constraints import PreTransformationConstraint

# from textattack.shared.utils import default_class_repr


class RepeatModification(PreTransformationConstraint):
    """A constraint disallowing the modification of words which have already
    been modified."""

    def _get_modifiable_indices(self, current_text):
        """Returns the word indices in current_text which are able to be
        deleted."""
        try:
            return (
                set(range(len(current_text.words)))
                - current_text.attack_attrs["modified_indices"]
            )
        except KeyError:
            raise KeyError(
                "`modified_indices` in attack_attrs required for RepeatModification constraint."
            )
