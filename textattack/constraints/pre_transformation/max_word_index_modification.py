"""

Max Word Index Modification
-----------------------------

"""
from textattack.constraints import PreTransformationConstraint
from textattack.shared import AttackedText
from typing import List, Set

# from textattack.shared.utils import default_class_repr


class MaxWordIndexModification(PreTransformationConstraint):
    """A constraint disallowing the modification of words which are past some
    maximum length limit."""

    def __init__(self, max_length: int):
        self.max_length = max_length

    def _get_modifiable_indices(self, current_text: AttackedText) -> Set[int]:
        """Returns the word indices in current_text which are able to be
        deleted."""
        return set(range(min(self.max_length, len(current_text.words))))

    def extra_repr_keys(self) -> List[str]:
        return ["max_length"]
