"""
Thought Vector
---------------------
"""

import functools

import torch

from textattack.shared import AbstractWordEmbedding, WordEmbedding, utils

from .sentence_encoder import SentenceEncoder


class ThoughtVector(SentenceEncoder):
    """A constraint on the distance between two sentences' thought vectors.

    Args:
        word_embedding (textattack.shared.AbstractWordEmbedding): The word embedding to use
    """

    def __init__(
        self, embedding=WordEmbedding.counterfitted_GLOVE_embedding(), **kwargs
    ):
        if not isinstance(embedding, AbstractWordEmbedding):
            raise ValueError(
                "`embedding` object must be of type `textattack.shared.AbstractWordEmbedding`."
            )
        self.word_embedding = embedding
        super().__init__(**kwargs)

    def clear_cache(self):
        self._get_thought_vector.cache_clear()

    @functools.lru_cache(maxsize=2 ** 10)
    def _get_thought_vector(self, text):
        """Sums the embeddings of all the words in ``text`` into a "thought
        vector"."""
        embeddings = []
        for word in utils.words_from_text(text):
            embedding = self.word_embedding[word]
            if embedding is not None:  # out-of-vocab words do not have embeddings
                embeddings.append(embedding)
        embeddings = torch.tensor(embeddings)
        return torch.mean(embeddings, dim=0)

    def encode(self, raw_text_list):
        return torch.stack([self._get_thought_vector(text) for text in raw_text_list])

    def extra_repr_keys(self):
        """Set the extra representation of the constraint using these keys."""
        return ["word_embedding"] + super().extra_repr_keys()
