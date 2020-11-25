"""

Edit Distance Constraints
--------------------------


"""

from typing import List

import editdistance

from textattack.constraints import Constraint
from textattack.shared import AttackedText


class LevenshteinEditDistance(Constraint):
    """A constraint on edit distance (Levenshtein Distance).

    Args:
        max_edit_distance (int): Maximum edit distance allowed.
        compare_against_original (bool):  If `True`, compare new `x_adv` against the original `x`.
            Otherwise, compare it against the previous `x_adv`.
    """

    def __init__(self, max_edit_distance: float, compare_against_original: bool = True):
        super().__init__(compare_against_original)
        if not isinstance(max_edit_distance, int):
            raise TypeError("max_edit_distance must be an int")
        self.max_edit_distance = max_edit_distance

    def _check_constraint(
        self, transformed_text: AttackedText, reference_text: AttackedText
    ) -> bool:
        edit_distance = editdistance.eval(reference_text.text, transformed_text.text)
        return edit_distance <= self.max_edit_distance

    def extra_repr_keys(self) -> List[str]:
        return ["max_edit_distance"] + super().extra_repr_keys()
