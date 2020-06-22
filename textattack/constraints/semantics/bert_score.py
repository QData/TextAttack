import bert_score
import nltk

from textattack.constraints import Constraint
from textattack.shared import utils


class BERTScore(Constraint):
    """ 
    A constraint on BERTScore difference. BERTScore is introduced in this paper 
        "BERTScore: Evaluating Text Generation with BERT" (Zhang et al, 2019)  https://arxiv.org/abs/1904.09675
    Args:
        min_bert_score (float): minimum threshold value for BERTScore
        model (str): name of model to use for scoring
        score_type (str): Pick one of three choices: (1) "precision", (2) "recall", (3) "f1"
    """

    def __init__(self, min_bert_score, model="bert-base-uncased", score_type="f1"):
        if not isinstance(min_bert_score, float):
            raise TypeError("max_bleu_score must be a float")
        if min_bert_score < 0.0 or min_bert_score > 1.0:
            raise ValueError("max_bert_score must be a value between 0.0 and 1.0")

        self.min_bert_score = min_bert_score
        self.model = model
        self.score_type = score_type
        # Turn off idf-weighting scheme b/c reference sentence set is small
        self._bert_scorer = bert_score.BERTScorer(
            model_type=model, idf=False, device=utils.device
        )
        self._score_type2idx = {"precision": 0, "recall": 1, "f1": 2}

    def _check_constraint(self, transformed_text, current_text, original_text=None):
        cand = transformed_text.text
        ref = original_text.text if original_text else current_text.text
        result = self._bert_scorer.score([cand], [ref])
        score = result[self._score_type2idx[self.score_type]].item()
        if score >= self.min_bert_score:
            return True
        else:
            return False

    def extra_repr_keys(self):
        return ["min_bert_score", "model", "score_type"]
