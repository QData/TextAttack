"""
Shared loads word embeddings and related distances
=====================================================
"""

from collections import defaultdict
import os
import pickle

import numpy as np
import torch

import textattack
from textattack.shared import utils


class WordEmbedding:
    """An object that loads word embeddings and related distances.

    Args:
        embedding_type (str): The type of the embedding to load automatically
        embedding_source (str): Source of embeddings provided,
        "defaults" corresponds to the textattack s3 bucket
        "gensim" expects a word2vec model
    """

    PATH = "word_embeddings"
    EMBEDDINGS_AVAILABLE_IN_TEXTATTACK = {"paragramcf"}

    def __init__(self, embedding_type=None, embedding_source=None):

        self.embedding_type = embedding_type or "paragramcf"
        self.embedding_source = embedding_source or "defaults"

        if self.embedding_source == "defaults":
            if (
                self.embedding_type
                not in WordEmbedding.EMBEDDINGS_AVAILABLE_IN_TEXTATTACK
            ):
                raise ValueError(
                    f"{self.embedding_type} is not available in TextAttack."
                )
        elif self.embedding_source not in ["gensim"]:
            raise ValueError(f"{self.embedding_source} type is not supported.")

        self._embeddings = None
        self._word2index = None
        self._index2word = None
        self._cos_sim_mat = None
        self._mse_dist_mat = None
        self._nn = None

        self._gensim_keyed_vectors = None

        self._init_embeddings_from_type(self.embedding_source, self.embedding_type)

    def _init_embeddings_from_defaults(self, embedding_type):
        """
        Init embeddings prepared in the textattack s3 bucket
        Args:
            embedding_type:

        Returns:

        """
        if embedding_type == "paragramcf":
            word_embeddings_folder = "paragramcf"
            word_embeddings_file = "paragram.npy"
            word_list_file = "wordlist.pickle"
            mse_dist_file = "mse_dist.p"
            cos_sim_file = "cos_sim.p"
            nn_matrix_file = "nn.npy"
        else:
            raise ValueError(f"Could not find word embedding {embedding_type}")

        # Download embeddings if they're not cached.
        word_embeddings_folder = os.path.join(
            WordEmbedding.PATH, word_embeddings_folder
        )
        word_embeddings_folder = textattack.shared.utils.download_if_needed(
            word_embeddings_folder
        )

        # Concatenate folder names to create full path to files.
        word_embeddings_file = os.path.join(
            word_embeddings_folder, word_embeddings_file
        )
        word_list_file = os.path.join(word_embeddings_folder, word_list_file)
        mse_dist_file = os.path.join(word_embeddings_folder, mse_dist_file)
        cos_sim_file = os.path.join(word_embeddings_folder, cos_sim_file)
        nn_matrix_file = os.path.join(word_embeddings_folder, nn_matrix_file)

        # loading the files
        self._embeddings = np.load(word_embeddings_file)
        self._word2index = np.load(word_list_file, allow_pickle=True)
        if os.path.exists(mse_dist_file):
            with open(mse_dist_file, "rb") as f:
                self._mse_dist_mat = pickle.load(f)
        else:
            self._mse_dist_mat = defaultdict(dict)
        if os.path.exists(cos_sim_file):
            with open(cos_sim_file, "rb") as f:
                self._cos_sim_mat = pickle.load(f)
        else:
            self._cos_sim_mat = defaultdict(dict)

        self._nn = np.load(nn_matrix_file)

        # Build glove dict and index.
        self._index2word = dict()
        for word, index in self._word2index.items():
            self._index2word[index] = word

    def _init_embeddings_from_gensim(self, embedding_type):
        """
        Initialize word embedding from a gensim word2vec model
        Args:
            embedding_type:

        Returns:

        """
        import gensim

        if embedding_type.endswith(".bin"):
            self._gensim_keyed_vectors = (
                gensim.models.KeyedVectors.load_word2vec_format(
                    embedding_type, binary=True
                )
            )
        else:
            self._gensim_keyed_vectors = (
                gensim.models.KeyedVectors.load_word2vec_format(embedding_type)
            )
        self._gensim_keyed_vectors.init_sims()

    def _init_embeddings_from_type(self, embedding_source, embedding_type):
        """Initializes embedding based on the source.

        Downloads and loads embeddings into memory.
        """

        if embedding_source == "defaults":
            self._init_embeddings_from_defaults(embedding_type)
        elif embedding_source == "gensim":
            self._init_embeddings_from_gensim(embedding_type)
        else:
            raise ValueError(f"Not supported word embedding source {embedding_source}")

    def __getitem__(self, index):
        """Gets a word embedding by word or ID.

        If word or ID not found, returns None.
        """
        if self.embedding_source == "defaults":
            if isinstance(index, str):
                try:
                    index = self._word2index[index]
                except KeyError:
                    return None
            try:
                return self._embeddings[index]
            except IndexError:
                # word embedding ID out of bounds
                return None
        elif self.embedding_source == "gensim":
            if isinstance(index, str):
                try:
                    index = self._gensim_keyed_vectors.vocab.get(index).index
                except KeyError:
                    return None
            try:
                return self._gensim_keyed_vectors.vectors_norm[index]
            except IndexError:
                # word embedding ID out of bounds
                return None
        else:
            raise ValueError(
                f"Not supported word embedding source {self.embedding_source}"
            )

    def get_cos_sim(self, a, b):
        """
        get cosine similarity of two words/IDs
        Args:
            a:
            b:

        Returns:

        """
        if self.embedding_source == "defaults":
            if isinstance(a, str):
                a = self._word2index[a]
            if isinstance(b, str):
                b = self._word2index[b]
            a, b = min(a, b), max(a, b)
            try:
                cos_sim = self._cos_sim_mat[a][b]
            except KeyError:
                e1 = self._embeddings[a]
                e2 = self._embeddings[b]
                e1 = torch.tensor(e1).to(utils.device)
                e2 = torch.tensor(e2).to(utils.device)
                cos_sim = torch.nn.CosineSimilarity(dim=0)(e1, e2).item()
                self._cos_sim_mat[a][b] = cos_sim
            return cos_sim
        elif self.embedding_source == "gensim":
            if not isinstance(a, str):
                a = self._gensim_keyed_vectors.index2word[a]
            if not isinstance(b, str):
                b = self._gensim_keyed_vectors.index2word[b]
            cos_sim = self._gensim_keyed_vectors.similarity(a, b)
            return cos_sim
        else:
            raise ValueError(
                f"Not supported word embedding source {self.embedding_source}"
            )

    def get_mse_dist(self, a, b):
        """
        get mse distance of two IDs
        Args:
            a:
            b:

        Returns:

        """
        if self.embedding_source == "defaults":
            a, b = min(a, b), max(a, b)
            try:
                mse_dist = self._mse_dist_mat[a][b]
            except KeyError:
                e1 = self._embeddings[a]
                e2 = self._embeddings[b]
                e1 = torch.tensor(e1).to(utils.device)
                e2 = torch.tensor(e2).to(utils.device)
                mse_dist = torch.sum((e1 - e2) ** 2).item()
                self._mse_dist_mat[a][b] = mse_dist
            return mse_dist
        elif self.embedding_source == "gensim":
            if self._mse_dist_mat is None:
                self._mse_dist_mat = defaultdict(dict)
            try:
                mse_dist = self._mse_dist_mat[a][b]
            except KeyError:
                e1 = self._gensim_keyed_vectors.vectors_norm[a]
                e2 = self._gensim_keyed_vectors.vectors_norm[b]
                e1 = torch.tensor(e1).to(utils.device)
                e2 = torch.tensor(e2).to(utils.device)
                mse_dist = torch.sum((e1 - e2) ** 2).item()
                self._mse_dist_mat[a][b] = mse_dist
            return mse_dist
        else:
            raise ValueError(
                f"Not supported word embedding source {self.embedding_source}"
            )

    def word2ind(self, word):
        """
        word to index
        Args:
            word:

        Returns:

        """
        if self.embedding_source == "defaults":
            return self._word2index[word]
        elif self.embedding_source == "gensim":
            vocab = self._gensim_keyed_vectors.vocab.get(word)
            if vocab is None:
                raise KeyError(word)
            return vocab.index
        else:
            raise ValueError(
                f"Not supported word embedding source {self.embedding_source}"
            )

    def ind2word(self, index):
        """
        index to word
        Args:
            index:

        Returns:

        """
        if self.embedding_source == "defaults":
            return self._index2word[index]
        elif self.embedding_source == "gensim":
            try:
                # this is a list, so the error would be IndexError
                return self._gensim_keyed_vectors.index2word[index]
            except IndexError:
                raise KeyError(index)
        else:
            raise ValueError(
                f"Not supported word embedding source {self.embedding_source}"
            )

    def nn(self, index, topn):
        """
        get top n nearest neighbours for a word
        Args:
            index:
            topn:

        Returns:

        """
        if self.embedding_source == "defaults":
            return self._nn[index][1 : (topn + 1)]
        elif self.embedding_source == "gensim":
            word = self._gensim_keyed_vectors.index2word[index]
            return [
                self._gensim_keyed_vectors.index2word.index(i[0])
                for i in self._gensim_keyed_vectors.similar_by_word(word, topn)
            ]
        else:
            raise ValueError(
                f"Not supported word embedding source {self.embedding_source}"
            )
