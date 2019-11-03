import numpy as np
import os

from textattack import utils as utils
from textattack.transformations.word_swap import WordSwap

from scipy.spatial.distance import cosine as scipy_cosine_dist

class WordSwapEmbedding(WordSwap):
    """ Transforms an input by replacing its words with synonyms in the word
        embedding space. """
    
    PATH = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/AttackGeneration/word_embeddings'
    
    def __init__(self, replace_stopwords=False, word_embedding='paragramcf', similarity_threshold=None):
        super().__init__(replace_stopwords)
        
        self.similarity_threshold = similarity_threshold
        
        if word_embedding == 'paragramcf':
            word_embeddings_folder = 'paragramcf'
            word_embeddings_file = 'paragram.npy'
            word_list_file = 'wordlist.pickle'
            nn_matrix_file = 'nn.npy'
            cos_dist_matrix_file = 'dist.npy'
        else:
            raise ValueError(f'Could not find word embedding {word_embedding}')
        
        # Download embeddings if they're not cached.
        utils.download_if_needed(WordSwapEmbedding.PATH)
        # Concatenate folder names to create full path to files.
        word_embeddings_file = os.path.join(WordSwapEmbedding.PATH, word_embeddings_folder, word_embeddings_file)
        word_list_file = os.path.join(WordSwapEmbedding.PATH, word_embeddings_folder, word_list_file)
        nn_matrix_file = os.path.join(WordSwapEmbedding.PATH, word_embeddings_folder, nn_matrix_file)
        cos_dist_matrix_file = os.path.join(WordSwapEmbedding.PATH, word_embeddings_folder, cos_dist_matrix_file)
        
        # Actually load the files from disk.
        self.word_embeddings = np.load(word_embeddings_file)
        self.word_embedding_word2index = np.load(word_list_file, allow_pickle=True)
        self.nn = np.load(nn_matrix_file)
        self.cos_dist_matrix = np.load(cos_dist_matrix_file)
        
        # Build glove dict and index.
        self.word_embedding_index2word = {}
        for word, index in self.word_embedding_word2index.items():
            self.word_embedding_index2word[index] = word
            
        self.name = "Word Swap Embedding"
    
    def _get_replacement_words(self, word, max_candidates=15):
        """ Returns a list of possible 'candidate words' to replace a word in a sentence 
            or phrase. Based on nearest neighbors selected word embeddings.
        """
        try:
            word_id = self.word_embedding_word2index[word]
            nnids = self.nn[word_id][1:max_candidates+1]
            candidate_words = []
            for i, nbr_id in enumerate(nnids):
                if self.similarity_threshold:
                    # Use precomputed cosine distances., if possible.
                    if self.cos_dist_matrix is not None:
                        cos_dist = self.cos_dist_matrix[word_id][i+1]
                    else:
                        e1 = self.word_embeddings[word_id]
                        e2 = self.word_embeddings[nbr_id]
                        cos_dist = scipy_cosine_dist(e1, e2)
                    cos_sim = 1-cos_dist
                    if cos_sim < self.similarity_threshold:
                        continue
                candidate_words.append(self.word_embedding_index2word[nbr_id])
            return candidate_words
        except KeyError:
            # This word is not in our word embedding database, so return an empty list.
            return []
