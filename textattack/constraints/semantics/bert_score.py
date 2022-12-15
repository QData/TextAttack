"""
BERT Score
---------------------
BERT Score is introduced in this paper (BERTScore: Evaluating Text Generation with BERT) `arxiv link`_.

.. _arxiv link: https://arxiv.org/abs/1904.09675

BERT Score measures token similarity between two text using contextual embedding.

To decide which two tokens to compare, it greedily chooses the most similar token from one text and matches it to a token in the second text.

"""

import bert_score

from textattack.constraints import Constraint
from textattack.shared import utils


class BERTScore(Constraint):
    """A constraint on BERT-Score difference.

    Args:
        min_bert_score (float), minimum threshold value for BERT-Score
        model_name (str), name of model to use for scoring
        num_layers (int), number of hidden layers in the model
        score_type (str), Pick one of following three choices

            -(1) ``precision`` : match words from candidate text to reference text
            -(2) ``recall`` :  match words from reference text to candidate text
            -(3) ``f1``: harmonic mean of precision and recall (recommended)

        compare_against_original (bool):
            If ``True``, compare new ``x_adv`` against the original ``x``.
            Otherwise, compare it against the previous ``x_adv``.
    """

    SCORE_TYPE2IDX = {"precision": 0, "recall": 1, "f1": 2}

    def __init__(
        self,
        min_bert_score,
        model_name="bert-base-uncased",
        num_layers=None,
        score_type="f1",
        compare_against_original=True,
    ):
        super().__init__(compare_against_original)
        if not isinstance(min_bert_score, float):
            raise TypeError("max_bert_score must be a float")
        if min_bert_score < 0.0 or min_bert_score > 1.0:
            raise ValueError("max_bert_score must be a value between 0.0 and 1.0")

        self.min_bert_score = min_bert_score
        self.model = model_name
        self.score_type = score_type
        # Turn off idf-weighting scheme b/c reference sentence set is small
        self._bert_scorer = bert_score.BERTScorer(
            model_type=model_name, idf=False, device=utils.device, num_layers=num_layers
        )

    def _sim_score(self, starting_text, transformed_text):
        """Returns the metric similarity between the embedding of the starting
        text and the transformed text.

        Args:
            starting_text: The ``AttackedText``to use as a starting point.
            transformed_text: A transformed ``AttackedText``

        Returns:
            The similarity between the starting and transformed text using BERTScore metric.
        """
        cand = transformed_text.text
        ref = starting_text.text
        result = self._bert_scorer.score([cand], [ref])
        return result[BERTScore.SCORE_TYPE2IDX[self.score_type]].item()

    def _check_constraint(self, transformed_text, reference_text):
        """Return `True` if BERT Score between `transformed_text` and
        `reference_text` is lower than minimum BERT Score."""
        score = self._sim_score(reference_text, transformed_text)
        if score >= self.min_bert_score:
            return True
        else:
            return False

    def extra_repr_keys(self):
        return ["min_bert_score", "model", "score_type"] + super().extra_repr_keys()
