import importlib
import os

import numpy as np
import pytest

from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.shared import AttackedText, GensimWordEmbedding, WordEmbedding

_gensim_available = importlib.util.find_spec("gensim") is not None


@pytest.fixture
def small_embedding():
    return WordEmbedding(
        np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 1.0]]),
        {"source": 0, "opposite": 1, "different": 2},
        {0: "source", 1: "opposite", 2: "different"},
    )


def test_word_embedding_distance_accepts_zero_thresholds(small_embedding):
    WordEmbeddingDistance(embedding=small_embedding, min_cos_sim=0.0)
    WordEmbeddingDistance(embedding=small_embedding, max_mse_dist=0.0)


@pytest.mark.parametrize(
    ("min_cos_sim", "max_mse_dist"),
    [(None, None), (0.0, 0.0), (0.5, 1.0)],
)
def test_word_embedding_distance_requires_exactly_one_threshold(
    small_embedding, min_cos_sim, max_mse_dist
):
    with pytest.raises(ValueError):
        WordEmbeddingDistance(
            embedding=small_embedding,
            min_cos_sim=min_cos_sim,
            max_mse_dist=max_mse_dist,
        )


def test_word_embedding_distance_enforces_zero_thresholds(small_embedding):
    reference = AttackedText("source")

    assert not WordEmbeddingDistance(
        embedding=small_embedding, min_cos_sim=0.0
    )._check_constraint(reference.generate_new_attacked_text(["opposite"]), reference)
    assert not WordEmbeddingDistance(
        embedding=small_embedding, max_mse_dist=0.0
    )._check_constraint(reference.generate_new_attacked_text(["different"]), reference)


def test_embedding_paragramcf():
    word_embedding = WordEmbedding.counterfitted_GLOVE_embedding()
    assert pytest.approx(word_embedding[0][0]) == -0.022007
    assert pytest.approx(word_embedding["fawn"][0]) == -0.022007
    assert word_embedding[10**9] is None


@pytest.mark.skipif(not _gensim_available, reason="gensim is not installed")
def test_embedding_gensim():
    # download a trained word2vec model
    from textattack.shared.utils.install import TEXTATTACK_CACHE_DIR

    path = os.path.join(TEXTATTACK_CACHE_DIR, "test_gensim_embedding.txt")
    f = open(path, "w")
    f.write(
        """4 2
hi 1 0
hello 1 1
bye -1 0
bye-bye -1 1
    """
    )
    f.close()

    from gensim.models import KeyedVectors

    keyed_vectors = KeyedVectors.load_word2vec_format(path)
    word_embedding = GensimWordEmbedding(keyed_vectors)
    assert pytest.approx(word_embedding[0][0]) == 1
    assert pytest.approx(word_embedding["bye-bye"][0]) == -1 / np.sqrt(2)
    assert word_embedding[10**9] is None

    # test query functionality
    assert pytest.approx(word_embedding.get_cos_sim(1, 3)) == 0
    # mse dist
    assert pytest.approx(word_embedding.get_mse_dist(0, 2)) == 4
    # nearest neighbour of hi is hello
    assert word_embedding.nearest_neighbours(0, 1)[0] == 1
    assert word_embedding.word2index("bye") == 2
    assert word_embedding.index2word(3) == "bye-bye"
    # remove test file
    os.remove(path)
