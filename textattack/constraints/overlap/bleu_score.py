"""

BLEU Constraints
--------------------------


"""

import nltk

from textattack.constraints import Constraint


class BLEU(Constraint):
    """A constraint on BLEU score difference.

    Args:
        max_bleu_score (int): Maximum BLEU score allowed.
        compare_against_original (bool):  If `True`, compare new `x_adv` against the original `x`.
            Otherwise, compare it against the previous `x_adv`.
    """

    def __init__(self, max_bleu_score, compare_against_original=True):
        super().__init__(compare_against_original)
        if not isinstance(max_bleu_score, int):
            raise TypeError("max_bleu_score must be an int")
        self.max_bleu_score = max_bleu_score

    def _check_constraint(self, transformed_text, reference_text):
        ref = reference_text.words
        hyp = transformed_text.words
        bleu_score = nltk.translate.bleu_score.sentence_bleu([ref], hyp)
        return bleu_score <= self.max_bleu_score

    def extra_repr_keys(self):
        return ["max_bleu_score"] + super().extra_repr_keys()
