"""
Shared loads word embeddings and related distances
=====================================================
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import os
import pickle

import numpy as np
import torch

from textattack.shared import utils


class AbstractWordEmbedding(ABC):
    """Abstract class representing word embedding used by TextAttack.

    This class specifies all the methods that is required to be defined
    so that it can be used for transformation and constraints. For
    custom word embedding not supported by TextAttack, please create a
    class that inherits this class and implement the required methods.
    However, please first check if you can use `WordEmbedding` class,
    which has a lot of internal methods implemented.
    """

    @abstractmethod
    def __getitem__(self, index):
        """Gets the embedding vector for word/id
        Args:
            index (Union[str|int]): `index` can either be word or integer representing the id of the word.
        Returns:
            vector (ndarray): 1-D embedding vector. If corresponding vector cannot be found for `index`, returns `None`.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_mse_dist(self, a, b):
        """Return MSE distance between vector for word `a` and vector for word
        `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a (Union[str|int]): Either word or integer presenting the id of the word
            b (Union[str|int]): Either word or integer presenting the id of the word
        Returns:
            distance (float): MSE (L2) distance
        """
        raise NotImplementedError()

    @abstractmethod
    def get_cos_sim(self, a, b):
        """Return cosine similarity between vector for word `a` and vector for
        word `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a (Union[str|int]): Either word or integer presenting the id of the word
            b (Union[str|int]): Either word or integer presenting the id of the word
        Returns:
            distance (float): cosine similarity
        """
        raise NotImplementedError()

    @abstractmethod
    def word2index(self, word):
        """
        Convert between word to id (i.e. index of word in embedding matrix)
        Args:
            word (str)
        Returns:
            index (int)
        """
        raise NotImplementedError()

    @abstractmethod
    def index2word(self, index):
        """
        Convert index to corresponding word
        Args:
            index (int)
        Returns:
            word (str)
        """
        raise NotImplementedError()

    @abstractmethod
    def nearest_neighbours(self, index, topn):
        """
        Get top-N nearest neighbours for a word
        Args:
            index (int): ID of the word for which we're finding the nearest neighbours
            topn (int): Used for specifying N nearest neighbours
        Returns:
            neighbours (list[int]): List of indices of the nearest neighbours
        """
        raise NotImplementedError()

    __repr__ = __str__ = utils.default_class_repr


class WordEmbedding(AbstractWordEmbedding):
    """Object for loading word embeddings and related distances for TextAttack.
    This class has a lot of internal components (e.g. get consine similarity)
    implemented. Consider using this class if you can provide the appropriate
    input data to create the object.

    Args:
        emedding_matrix (ndarray): 2-D array of shape N x D where N represents size of vocab and D is the dimension of embedding vectors.
        word2index (Union[dict|object]): dictionary (or a similar object) that maps word to its index with in the embedding matrix.
        index2word (Union[dict|object]): dictionary (or a similar object) that maps index to its word.
        nn_matrix (ndarray): Matrix for precomputed nearest neighbours. It should be a 2-D integer array of shape N x K
            where N represents size of vocab and K is the top-K nearest neighbours. If this is set to `None`, we have to compute nearest neighbours
            on the fly for `nearest_neighbours` method, which is costly.
    """

    PATH = "word_embeddings"

    def __init__(self, embedding_matrix, word2index, index2word, nn_matrix=None):
        self.embedding_matrix = embedding_matrix
        self._word2index = word2index
        self._index2word = index2word
        self.nn_matrix = nn_matrix

        # Dictionary for caching results
        self._mse_dist_mat = defaultdict(dict)
        self._cos_sim_mat = defaultdict(dict)
        self._nn_cache = {}

    def __getitem__(self, index):
        """Gets the embedding vector for word/id
        Args:
            index (Union[str|int]): `index` can either be word or integer representing the id of the word.
        Returns:
            vector (ndarray): 1-D embedding vector. If corresponding vector cannot be found for `index`, returns `None`.
        """
        if isinstance(index, str):
            try:
                index = self._word2index[index]
            except KeyError:
                return None
        try:
            return self.embedding_matrix[index]
        except IndexError:
            # word embedding ID out of bounds
            return None

    def word2index(self, word):
        """
        Convert between word to id (i.e. index of word in embedding matrix)
        Args:
            word (str)
        Returns:
            index (int)
        """
        return self._word2index[word]

    def index2word(self, index):
        """
        Convert index to corresponding word
        Args:
            index (int)
        Returns:
            word (str)

        """
        return self._index2word[index]

    def get_mse_dist(self, a, b):
        """Return MSE distance between vector for word `a` and vector for word
        `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a (Union[str|int]): Either word or integer presenting the id of the word
            b (Union[str|int]): Either word or integer presenting the id of the word
        Returns:
            distance (float): MSE (L2) distance
        """
        if isinstance(a, str):
            a = self._word2index[a]
        if isinstance(b, str):
            b = self._word2index[b]
        a, b = min(a, b), max(a, b)
        try:
            mse_dist = self._mse_dist_mat[a][b]
        except KeyError:
            e1 = self.embedding_matrix[a]
            e2 = self.embedding_matrix[b]
            e1 = torch.tensor(e1).to(utils.device)
            e2 = torch.tensor(e2).to(utils.device)
            mse_dist = torch.sum((e1 - e2) ** 2).item()
            self._mse_dist_mat[a][b] = mse_dist

        return mse_dist

    def get_cos_sim(self, a, b):
        """Return cosine similarity between vector for word `a` and vector for
        word `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a (Union[str|int]): Either word or integer presenting the id of the word
            b (Union[str|int]): Either word or integer presenting the id of the word
        Returns:
            distance (float): cosine similarity
        """
        if isinstance(a, str):
            a = self._word2index[a]
        if isinstance(b, str):
            b = self._word2index[b]
        a, b = min(a, b), max(a, b)
        try:
            cos_sim = self._cos_sim_mat[a][b]
        except KeyError:
            e1 = self.embedding_matrix[a]
            e2 = self.embedding_matrix[b]
            e1 = torch.tensor(e1).to(utils.device)
            e2 = torch.tensor(e2).to(utils.device)
            cos_sim = torch.nn.CosineSimilarity(dim=0)(e1, e2).item()
            self._cos_sim_mat[a][b] = cos_sim
        return cos_sim

    def nearest_neighbours(self, index, topn):
        """
        Get top-N nearest neighbours for a word
        Args:
            index (int): ID of the word for which we're finding the nearest neighbours
            topn (int): Used for specifying N nearest neighbours
        Returns:
            neighbours (list[int]): List of indices of the nearest neighbours
        """
        if isinstance(index, str):
            index = self._word2index[index]
        if self.nn_matrix is not None:
            nn = self.nn_matrix[index][1 : (topn + 1)]
        else:
            try:
                nn = self._nn_cache[index]
            except KeyError:
                embedding = torch.tensor(self.embedding_matrix).to(utils.device)
                vector = torch.tensor(self.embedding_matrix[index]).to(utils.device)
                dist = torch.norm(embedding - vector, dim=1, p=None)
                # Since closest neighbour will be the same word, we consider N+1 nearest neighbours
                nn = dist.topk(topn + 1, largest=False)[1:].tolist()
                self._nn_cache[index] = nn

        return nn

    @staticmethod
    def counterfitted_GLOVE_embedding():
        """Returns a prebuilt counter-fitted GLOVE word embedding proposed by
        "Counter-fitting Word Vectors to Linguistic Constraints" (Mrkšić et
        al., 2016)"""
        if (
            "textattack_counterfitted_GLOVE_embedding" in utils.GLOBAL_OBJECTS
            and isinstance(
                utils.GLOBAL_OBJECTS["textattack_counterfitted_GLOVE_embedding"],
                WordEmbedding,
            )
        ):
            # avoid recreating same embedding (same memory) and instead share across different components
            return utils.GLOBAL_OBJECTS["textattack_counterfitted_GLOVE_embedding"]

        word_embeddings_folder = "paragramcf"
        word_embeddings_file = "paragram.npy"
        word_list_file = "wordlist.pickle"
        mse_dist_file = "mse_dist.p"
        cos_sim_file = "cos_sim.p"
        nn_matrix_file = "nn.npy"

        # Download embeddings if they're not cached.
        word_embeddings_folder = os.path.join(
            WordEmbedding.PATH, word_embeddings_folder
        )
        word_embeddings_folder = utils.download_if_needed(word_embeddings_folder)
        # Concatenate folder names to create full path to files.
        word_embeddings_file = os.path.join(
            word_embeddings_folder, word_embeddings_file
        )
        word_list_file = os.path.join(word_embeddings_folder, word_list_file)
        mse_dist_file = os.path.join(word_embeddings_folder, mse_dist_file)
        cos_sim_file = os.path.join(word_embeddings_folder, cos_sim_file)
        nn_matrix_file = os.path.join(word_embeddings_folder, nn_matrix_file)

        # loading the files
        embedding_matrix = np.load(word_embeddings_file)
        word2index = np.load(word_list_file, allow_pickle=True)
        index2word = {}
        for word, index in word2index.items():
            index2word[index] = word
        nn_matrix = np.load(nn_matrix_file)

        embedding = WordEmbedding(embedding_matrix, word2index, index2word, nn_matrix)

        with open(mse_dist_file, "rb") as f:
            mse_dist_mat = pickle.load(f)
        with open(cos_sim_file, "rb") as f:
            cos_sim_mat = pickle.load(f)

        embedding._mse_dist_mat = mse_dist_mat
        embedding._cos_sim_mat = cos_sim_mat

        utils.GLOBAL_OBJECTS["textattack_counterfitted_GLOVE_embedding"] = embedding

        return embedding


class GensimWordEmbedding(AbstractWordEmbedding):
    """Wraps Gensim's `models.keyedvectors` module
    (https://radimrehurek.com/gensim/models/keyedvectors.html)"""

    def __init__(self, keyed_vectors):
        gensim = utils.LazyLoader("gensim", globals(), "gensim")

        if isinstance(
            keyed_vectors, gensim.models.keyedvectors.WordEmbeddingsKeyedVectors
        ):
            self.keyed_vectors = keyed_vectors
        else:
            raise ValueError(
                "`keyed_vectors` argument must be a "
                "`gensim.models.keyedvectors.WordEmbeddingsKeyedVectors` object"
            )

        self.keyed_vectors.init_sims()
        self._mse_dist_mat = defaultdict(dict)
        self._cos_sim_mat = defaultdict(dict)

    def __getitem__(self, index):
        """Gets the embedding vector for word/id
        Args:
            index (Union[str|int]): `index` can either be word or integer representing the id of the word.
        Returns:
            vector (ndarray): 1-D embedding vector. If corresponding vector cannot be found for `index`, returns `None`.
        """
        if isinstance(index, str):
            try:
                index = self.keyed_vectors.vocab.get(index).index
            except KeyError:
                return None
        try:
            return self.keyed_vectors.vectors_norm[index]
        except IndexError:
            # word embedding ID out of bounds
            return None

    def word2index(self, word):
        """
        Convert between word to id (i.e. index of word in embedding matrix)
        Args:
            word (str)
        Returns:
            index (int)
        """
        vocab = self.keyed_vectors.vocab.get(word)
        if vocab is None:
            raise KeyError(word)
        return vocab.index

    def index2word(self, index):
        """
        Convert index to corresponding word
        Args:
            index (int)
        Returns:
            word (str)

        """
        try:
            # this is a list, so the error would be IndexError
            return self.keyed_vectors.index2word[index]
        except IndexError:
            raise KeyError(index)

    def get_mse_dist(self, a, b):
        """Return MSE distance between vector for word `a` and vector for word
        `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a (Union[str|int]): Either word or integer presenting the id of the word
            b (Union[str|int]): Either word or integer presenting the id of the word
        Returns:
            distance (float): MSE (L2) distance
        """
        try:
            mse_dist = self._mse_dist_mat[a][b]
        except KeyError:
            e1 = self.keyed_vectors.vectors_norm[a]
            e2 = self.keyed_vectors.vectors_norm[b]
            e1 = torch.tensor(e1).to(utils.device)
            e2 = torch.tensor(e2).to(utils.device)
            mse_dist = torch.sum((e1 - e2) ** 2).item()
            self._mse_dist_mat[a][b] = mse_dist
        return mse_dist

    def get_cos_sim(self, a, b):
        """Return cosine similarity between vector for word `a` and vector for
        word `b`.

        Since this is a metric, `get_mse_dist(a,b)` and `get_mse_dist(b,a)` should return the same value.
        Args:
            a (Union[str|int]): Either word or integer presenting the id of the word
            b (Union[str|int]): Either word or integer presenting the id of the word
        Returns:
            distance (float): cosine similarity
        """
        if not isinstance(a, str):
            a = self.keyed_vectors.index2word[a]
        if not isinstance(b, str):
            b = self.keyed_vectors.index2word[b]
        cos_sim = self.keyed_vectors.similarity(a, b)
        return cos_sim

    def nearest_neighbours(self, index, topn, return_words=True):
        """
        Get top-N nearest neighbours for a word
        Args:
            index (int): ID of the word for which we're finding the nearest neighbours
            topn (int): Used for specifying N nearest neighbours
        Returns:
            neighbours (list[int]): List of indices of the nearest neighbours
        """
        word = self.keyed_vectors.index2word[index]
        return [
            self.keyed_vectors.index2word.index(i[0])
            for i in self.keyed_vectors.similar_by_word(word, topn)
        ]
