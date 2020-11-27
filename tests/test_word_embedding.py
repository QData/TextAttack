import os

import numpy as np
import pytest

from textattack.shared import GensimWordEmbedding, WordEmbedding


def test_embedding_paragramcf():
    word_embedding = WordEmbedding.counterfitted_GLOVE_embedding()
    assert pytest.approx(word_embedding[0][0]) == -0.022007
    assert pytest.approx(word_embedding["fawn"][0]) == -0.022007
    assert word_embedding[10 ** 9] is None


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
    word_embedding = GensimWordEmbedding(path)
    assert pytest.approx(word_embedding[0][0]) == 1
    assert pytest.approx(word_embedding["bye-bye"][0]) == -1 / np.sqrt(2)
    assert word_embedding[10 ** 9] is None

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
