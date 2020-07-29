import functools

import nltk

import textattack

from .text_to_text_goal_function import TextToTextGoalFunction


class MinimizeBleu(TextToTextGoalFunction):
    """Attempts to minimize the BLEU score between the current output
    translation and the reference translation.

    BLEU score was defined in ``BLEU: a Method for Automatic Evaluation of
    Machine Translation''
    (https://www.aclweb.org/anthology/P02-1040.pdf).

    This goal function is defined in ``It’s Morphin’ Time! Combating
    Linguistic Discrimination with Inflectional Perturbations''.
    (https://www.aclweb.org/anthology/2020.acl-main.263/).
    """

    EPS = 1e-10

    def __init__(self, *args, target_bleu=0.0, **kwargs):
        self.target_bleu = target_bleu
        super().__init__(*args, **kwargs)

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()
        get_bleu.cache_clear()

    def _is_goal_complete(self, model_output, _):
        bleu_score = 1.0 - self._get_score(model_output, _)
        return bleu_score <= (self.target_bleu + MinimizeBleu.EPS)

    def _get_score(self, model_output, _):
        model_output_at = textattack.shared.AttackedText(model_output)
        ground_truth_at = textattack.shared.AttackedText(self.ground_truth_output)
        bleu_score = get_bleu(model_output_at, ground_truth_at)
        return 1.0 - bleu_score

    def extra_repr_keys(self):
        if self.maximizable:
            return ["maximizable"]
        else:
            return ["maximizable", "target_bleu"]


@functools.lru_cache(maxsize=2 ** 12)
def get_bleu(a, b):
    ref = a.words
    hyp = b.words
    bleu_score = nltk.translate.bleu_score.sentence_bleu([ref], hyp)
    return bleu_score
