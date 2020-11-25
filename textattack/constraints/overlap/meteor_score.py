"""

METEOR Constraints
--------------------------


"""


from typing import List

import nltk

from textattack.constraints import Constraint
from textattack.shared import AttackedText


class METEOR(Constraint):
    """A constraint on METEOR score difference.

    Args:
        max_meteor (int): Max METEOR score allowed.
        compare_against_original (bool):  If `True`, compare new `x_adv` against the original `x`.
            Otherwise, compare it against the previous `x_adv`.
    """

    def __init__(self, max_meteor: float, compare_against_original: bool = True):
        super().__init__(compare_against_original)
        if not isinstance(max_meteor, int):
            raise TypeError("max_meteor must be an int")
        self.max_meteor = max_meteor

    def _check_constraint(
        self, transformed_text: AttackedText, reference_text: AttackedText
    ):
        meteor = nltk.translate.meteor([reference_text], transformed_text)
        return meteor <= self.max_meteor

    def extra_repr_keys(self) -> List[str]:
        return ["max_meteor"] + super().extra_repr_keys()
