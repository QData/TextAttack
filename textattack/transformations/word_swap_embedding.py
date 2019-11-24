import numpy as np
import os
import torch

from textattack import utils as utils
from textattack.transformations.word_swap import WordSwap

class WordSwapEmbedding(WordSwap):
    """ Transforms an input by replacing its words with synonyms in the word
        embedding space. """
    
    PATH = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/AttackGeneration/word_embeddings'
    
    def __init__(self, max_candidates=15, replace_stopwords=False, 
        word_embedding='paragramcf', min_cos_sim=None, max_mse_dist=None, 
        **kwargs):
            
        super().__init__(**kwargs)
        
        self.max_candidates = max_candidates
        self.min_cos_sim = min_cos_sim
        self.max_mse_dist = max_mse_dist
        
        if word_embedding == 'paragramcf':
            word_embeddings_folder = 'paragramcf'
            word_embeddings_file = 'paragram.npy'
            word_list_file = 'wordlist.pickle'
            nn_matrix_file = 'nn.npy'
            cos_dist_matrix_file = 'cos_dist.npy'
            mse_dist_matrix_file = 'mse_dist.npy'
        else:
            raise ValueError(f'Could not find word embedding {word_embedding}')
        
        # Download embeddings if they're not cached.
        utils.download_if_needed(WordSwapEmbedding.PATH)
        # Concatenate folder names to create full path to files.
        word_embeddings_file = os.path.join(WordSwapEmbedding.PATH, word_embeddings_folder, word_embeddings_file)
        word_list_file = os.path.join(WordSwapEmbedding.PATH, word_embeddings_folder, word_list_file)
        nn_matrix_file = os.path.join(WordSwapEmbedding.PATH, word_embeddings_folder, nn_matrix_file)
        cos_dist_matrix_file = os.path.join(WordSwapEmbedding.PATH, word_embeddings_folder, cos_dist_matrix_file)
        mse_dist_matrix_file = os.path.join(WordSwapEmbedding.PATH, word_embeddings_folder, cos_dist_matrix_file)
        
        # Actually load the files from disk.
        self.word_embeddings = np.load(word_embeddings_file)
        self.word_embedding_word2index = np.load(word_list_file, allow_pickle=True)
        self.nn = np.load(nn_matrix_file)
        # @TODO don't load this file if it doesn't exist on disk.
        self.cos_dist_matrix = np.load(cos_dist_matrix_file)
        self.mse_dist_matrix = np.load(mse_dist_matrix_file)
        
        # Build glove dict and index.
        self.word_embedding_index2word = {}
        for word, index in self.word_embedding_word2index.items():
            self.word_embedding_index2word[index] = word
    
    def _vectors_meet_thresholds(self, word_id, nbr_id, nbr_nn_pos):
        """ Determines whether (v1, v2) are closer than `self.min_cos_sim`
            and `self.max_mse_dist`. """
        e1 = self.word_embeddings[word_id]
        e2 = self.word_embeddings[nbr_id]
        # Check cosine distance.
        if self.min_cos_sim:
            # Use precomputed cosine distances, if possible.
            if self.cos_dist_matrix is not None and word_id in self.cos_dist_matrix:
                cos_sim = 1 - self.cos_dist_matrix[word_id][nbr_nn_pos]
            else:
                e1 = torch.tensor(e1).to(utils.get_device())
                e2 = torch.tensor(e2).to(utils.get_device())
                cos_sim = torch.nn.CosineSimilarity(dim=0)(e1, e2)
            # print((word_id, nbr_id), 'cos_sim:', cos_sim)
            if cos_sim < self.min_cos_sim:
                return False
        # Check MSE distance.
        if self.max_mse_dist:
            if self.mse_dist_matrix is not None:
                mse_dist = self.mse_dist_matrix[word_id][nbr_nn_pos]
            else:
                mse_dist = torch.sum((e1 - e2) ** 2)
            if mse_dist > self.max_mse_dist:
                return False
        return True
        
    def _get_replacement_words(self, word):
        """ Returns a list of possible 'candidate words' to replace a word in a sentence 
            or phrase. Based on nearest neighbors selected word embeddings.
        """
        try:
            word_id = self.word_embedding_word2index[word.lower()]
            nnids = self.nn[word_id][1:self.max_candidates+1]
            candidate_words = []
            # @TODO vectorize.
            for i, nbr_id in enumerate(nnids):
                if self._vectors_meet_thresholds(word_id, nbr_id, i+1):
                    candidate_words.append(self.word_embedding_index2word[nbr_id])
            return candidate_words
        except KeyError:
            # This word is not in our word embedding database, so return an empty list.
            return []