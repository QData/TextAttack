"""
BERT for Sentence Similarity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

from textattack.constraints.semantics.sentence_encoders import SentenceEncoder
from textattack.shared import utils

sentence_transformers = utils.LazyLoader(
    "sentence_transformers", globals(), "sentence_transformers"
)


class BERT(SentenceEncoder):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using BERT, trained on NLI data, and
    fine- tuned on the STS benchmark dataset.

    Available models can be found here: https://huggingface.co/sentence-transformers
    """

    def __init__(
        self,
        threshold=0.7,
        metric="cosine",
        model_name="bert-base-nli-stsb-mean-tokens",
        **kwargs
    ):
        super().__init__(threshold=threshold, metric=metric, **kwargs)
        self.model = sentence_transformers.SentenceTransformer(model_name)
        self.model.to(utils.device)

    def encode(self, sentences):
        return self.model.encode(sentences)
