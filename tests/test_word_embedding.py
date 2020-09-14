import numpy as np
import pytest

from textattack.shared import WordEmbedding


def test_embedding_paragramcf():
    word_embedding = WordEmbedding(embedding_type="paragramcf")
    assert pytest.approx(word_embedding[0][0]) == -0.022007
    assert pytest.approx(word_embedding["fawn"][0]) == -0.022007
    assert word_embedding[10 ** 9] is None


def test_embedding_or_type_required():
    with pytest.raises(ValueError):
        WordEmbedding(embedding_type=None, embeddings=None)


def test_embedding_custom_lookup():
    embs = np.array([[0, 1, 2], [3, 4, 5]])
    word2index = {"hello": 0, "world": 1}
    word_embedding = WordEmbedding(embeddings=embs, word2index=word2index)
    assert pytest.approx(word_embedding[1][0]) == 3
    assert pytest.approx(word_embedding["world"][0]) == 3
    assert word_embedding[10 ** 9] is None
