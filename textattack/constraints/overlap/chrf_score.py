import nltk.translate.chrf_score

from textattack.constraints import Constraint


class chrF(Constraint):
    """A constraint on chrF (n-gram F-score) difference.

    Args:
        max_chrf (int): Max n-gram F-score allowed.
        compare_against_original (bool): If `True`, compare new `x_adv` against the original `x`.
            Otherwise, compare it against the previous `x_adv`.
    """

    def __init__(self, max_chrf, compare_against_original=True):
        super().__init__(compare_against_original)
        if not isinstance(max_chrf, int):
            raise TypeError("max_chrf must be an int")
        self.max_chrf = max_chrf

    def _check_constraint(self, transformed_text, reference_text):
        ref = reference_text.words
        hyp = transformed_text.words
        chrf = nltk.translate.chrf_score.sentence_chrf(ref, hyp)
        return chrf <= self.max_chrf

    def extra_repr_keys(self):
        return ["max_chrf"] + super().extra_repr_keys()
