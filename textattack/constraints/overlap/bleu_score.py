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
        if not (isinstance(max_bleu_score, int) or isinstance(max_bleu_score, float)):
            raise TypeError(
                f"max_bleu_score must be an int or float, got type {type(max_bleu_score)}"
            )
        self.max_bleu_score = max_bleu_score

    def _score(self, transformed_text, reference_text):
        ref = reference_text.words
        hyp = transformed_text.words
        return nltk.translate.bleu_score.sentence_bleu([ref], hyp)

    def _check_constraint(self, transformed_text, reference_text):
        bleu_score = self._score(transformed_text, reference_text)
        return bleu_score <= self.max_bleu_score

    def extra_repr_keys(self):
        return ["max_bleu_score"] + super().extra_repr_keys()
