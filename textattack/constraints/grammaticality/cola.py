"""
CoLA for Grammaticality
--------------------------

"""
import lru
import nltk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from textattack.constraints import Constraint
from textattack.models.wrappers import HuggingFaceModelWrapper


class COLA(Constraint):
    """Constrains an attack to text that has a similar number of linguistically
    accecptable sentences as the original text. Linguistic acceptability is
    determined by a model pre-trained on the `CoLA dataset <https://nyu-
    mll.github.io/CoLA/>`_. By default a BERT model is used, see the `pre-
    trained models README <https://github.com/QData/TextAttack/tree/master/
    textattack/models>`_ for a full list of available models or provide your
    own model from the huggingface model hub.

    Args:
        max_diff (float or int): The absolute (if int or greater than or equal to 1) or percent (if float and less than 1)
            maximum difference allowed between the number of valid sentences in the reference
            text and the number of valid sentences in the attacked text.
        model_name (str): The name of the pre-trained model to use for classification. The model must be in huggingface model hub.
        compare_against_original (bool): If `True`, compare against the original text.
            Otherwise, compare against the most recent text.
    """

    def __init__(
        self,
        max_diff,
        model_name="textattack/bert-base-uncased-CoLA",
        compare_against_original=True,
    ):
        super().__init__(compare_against_original)
        if not isinstance(max_diff, float) and not isinstance(max_diff, int):
            raise TypeError("max_diff must be a float or int")
        if max_diff < 0.0:
            raise ValueError("max_diff must be a value greater or equal to than 0.0")

        self.max_diff = max_diff
        self.model_name = model_name
        self._reference_score_cache = lru.LRU(2**10)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = HuggingFaceModelWrapper(model, tokenizer)

    def clear_cache(self):
        self._reference_score_cache.clear()

    def _check_constraint(self, transformed_text, reference_text):
        if reference_text not in self._reference_score_cache:
            # Split the text into sentences before predicting validity
            reference_sentences = nltk.sent_tokenize(reference_text.text)
            # A label of 1 indicates the sentence is valid
            num_valid = self.model(reference_sentences).argmax(axis=1).sum()
            self._reference_score_cache[reference_text] = num_valid

        sentences = nltk.sent_tokenize(transformed_text.text)
        predictions = self.model(sentences)
        num_valid = predictions.argmax(axis=1).sum()
        reference_score = self._reference_score_cache[reference_text]

        if isinstance(self.max_diff, int) or self.max_diff >= 1:
            threshold = reference_score - self.max_diff
        else:
            threshold = reference_score - (reference_score * self.max_diff)

        if num_valid < threshold:
            return False
        return True

    def extra_repr_keys(self):
        return [
            "max_diff",
            "model_name",
        ] + super().extra_repr_keys()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_reference_score_cache"] = self._reference_score_cache.get_size()
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._reference_score_cache = lru.LRU(state["_reference_score_cache"])
