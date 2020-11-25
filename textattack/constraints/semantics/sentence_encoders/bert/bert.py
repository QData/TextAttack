"""
BERT for Sentence Similarity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

from typing import List

import torch

from textattack.constraints.semantics.sentence_encoders import SentenceEncoder
from textattack.shared import utils

sentence_transformers = utils.LazyLoader(
    "sentence_transformers", globals(), "sentence_transformers"
)


class BERT(SentenceEncoder):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using BERT, trained on NLI data, and
    fine- tuned on the STS benchmark dataset."""

    def __init__(self, threshold: float = 0.7, metric: str = "cosine", **kwargs):
        super().__init__(threshold=threshold, metric=metric, **kwargs)
        self.model = sentence_transformers.SentenceTransformer(
            "bert-base-nli-stsb-mean-tokens"
        )
        self.model.to(utils.device)

    def encode(self, sentences: List[str]) -> torch.Tensor:
        return self.model.encode(sentences)
