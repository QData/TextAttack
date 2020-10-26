"""
Shared loads word embeddings and related distances
=====================================================
"""

import os
import pickle

import numpy as np

import textattack


class WordEmbedding:
    """An object that loads word embeddings and related distances.

    Args:
        embedding_type (str): The type of the embedding to load automatically
        embeddings: A dictionary or matrix that maps word embeddings from
            their IDs to vectors (torch tensors or numpy ndarrays). Must be
            provided for custom embeddings, when embedding_type is not provided.
            If both `embedding_type` and `embeddings` are provided, `embeddings`
            overrides `embedding_type`.
        word2index: A dictionary that maps words by string to ID. Not required,
            but useful if the user intends to use this class to look up word
            embeddings by their string. Can omit this argument and solely query
            words by ID from `self.embeddings`.
    """

    PATH = "word_embeddings"

    def __init__(self, embedding_type="paragramcf", embeddings=None, word2index=None):
        if embeddings is not None:
            self.embeddings = embeddings
            self.word2index = word2index or {}
        elif embedding_type:
            self._init_embeddings_from_type(embedding_type)
        else:
            raise ValueError(
                "Must supply `embedding_type` or `embeddings` as parameters."
            )

    def _init_embeddings_from_type(self, embedding_type):
        """Initializes self.embeddings based on the type of embedding.

        Downloads and loads embeddings into memory.
        """
        self.embedding_type = embedding_type
        if embedding_type == "paragramcf":
            word_embeddings_folder = "paragramcf"
            word_embeddings_file = "paragram.npy"
            word_list_file = "wordlist.pickle"
            mse_dist_file = "mse_dist.p"
            cos_sim_file = "cos_sim.p"
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

        # Actually load the files from disk.
        self.embeddings = np.load(word_embeddings_file)
        self.word2index = np.load(word_list_file, allow_pickle=True)

        # Precomputed distance matrices store distances at mat[x][y], where
        # x and y are word IDs and x < y.
        if os.path.exists(mse_dist_file):
            with open(mse_dist_file, "rb") as f:
                self.mse_dist_mat = pickle.load(f)
        else:
            self.mse_dist_mat = {}
        if os.path.exists(cos_sim_file):
            with open(cos_sim_file, "rb") as f:
                self.cos_sim_mat = pickle.load(f)
        else:
            self.cos_sim_mat = {}

    def __getitem__(self, index):
        """Gets a word embedding by word or ID.

        If word or ID not found, returns None.
        """
        if isinstance(index, str):
            try:
                index = self.word2index[index]
            except KeyError:
                return None
        try:
            return self.embeddings[index]
        except IndexError:
            # word embedding ID out of bounds
            return None
