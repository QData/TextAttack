import numpy as np
import os
import torch

from textattack import utils as utils
from textattack.constraints import Constraint
from textattack.tokenized_text import TokenizedText

class WordEmbeddingDistance(Constraint):
    """
    todo document here
    
    Params:
        word_embedding (str): The word embedding to use
        include_unknown_words (bool): Whether or not C(x,x_adv) is true
            if the embedding of x or x_adv is unknown
        min_cos_sim: the minimum cosine similarity between word embeddings
        max_mse_dist: the maximum euclidean distance between word embeddings
        embedding_cased (bool): whether embedding supports uppercase & lowercase
            (defaults to False, or just lowercase)
    """
    PATH = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/AttackGeneration/word_embeddings'
    def __init__(self, word_embedding='paragramcf', include_unknown_words=True,
        min_cos_sim=None, max_mse_dist=None, embedding_cased=False):
        self.include_unknown_words = include_unknown_words
        self.embedding_cased = embedding_cased
        self.min_cos_sim = min_cos_sim
        self.max_mse_dist = max_mse_dist
        
        if word_embedding == 'paragramcf':
            word_embeddings_folder = 'paragramcf'
            word_embeddings_file = 'paragram.npy'
            word_list_file = 'wordlist.pickle'
        else:
            raise ValueError(f'Could not find word embedding {word_embedding}')

        # Download embeddings if they're not cached.
        utils.download_if_needed(WordEmbeddingDistance.PATH)
        
        # Concatenate folder names to create full path to files.
        word_embeddings_file = os.path.join(WordEmbeddingDistance.PATH, word_embeddings_folder, word_embeddings_file)
        word_list_file = os.path.join(WordEmbeddingDistance.PATH, word_embeddings_folder, word_list_file)
        
        # Actually load the files from disk.
        # @TODO for each word embedding, precompute a single distance matrix
        # (for each distance metric) that maps _string_ word pairs to their
        # embedding distances.
        self.word_embeddings = np.load(word_embeddings_file)
        self.word_embedding_word2index = np.load(word_list_file, allow_pickle=True)
    
    def call_many(self, x, x_adv_list, original_word=None):
        """ Returns each `x_adv` from `x_adv_list` where `C(x,x_adv)` is True. 
        """
        return [x_adv for x_adv in x_adv_list if self(x, x_adv)]
    
    def __call__(self, x, x_adv):
        """ Returns true if (x, x_adv) are closer than `self.min_cos_sim`
            and `self.max_mse_dist`. """
        
        if not isinstance(x, TokenizedText):
            raise TypeError('x must be of type TokenizedText')
        if not isinstance(x_adv, TokenizedText):
            raise TypeError('x_adv must be of type TokenizedText')
        
        try:
            i = x_adv.attack_attrs['modified_word_index']
            x = x.words()[i]
            x_adv = x_adv.words()[i]
        except AttributeError:
            raise AttributeError('Cannot apply word embedding distance constraint without `modified_word_index`')
            
        if not self.embedding_cased:
            # If embedding vocabulary is all lowercase, lowercase words.
            x = x.lower()
            x_adv = x_adv.lower()
        
        try:
            x_id = self.word_embedding_word2index[x]
            x_adv_id = self.word_embedding_word2index[x_adv]
        except KeyError:
            # This error is thrown if x or x_adv has no corresponding ID.
            return self.include_unknown_words
            
        e1 = self.word_embeddings[x_id]
        e2 = self.word_embeddings[x_adv_id]
        # Convert to tensor.
        if self.min_cos_sim or self.max_mse_dist:
            e1 = torch.tensor(e1).to(utils.get_device())
            e2 = torch.tensor(e2).to(utils.get_device())
        # Check cosine distance.
        if self.min_cos_sim:
            cos_sim = torch.nn.CosineSimilarity(dim=0)(e1, e2)
            if cos_sim < self.min_cos_sim:
                return False
        # Check MSE distance.
        if self.max_mse_dist:
            mse_dist = torch.sum((e1 - e2) ** 2)
            if mse_dist > self.max_mse_dist:
                return False
        return True
        