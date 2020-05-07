import functools
import torch

from textattack.shared import utils
from textattack.constraints import Constraint
from textattack.shared import WordEmbedding
from textattack.shared import TokenizedText

class ThoughtVector(Constraint):
    """
    A constraint on the distance between two sentences' thought vectors.
    
    Args:
        word_embedding (str): The word embedding to use
        min_cos_sim: the minimum cosine similarity between thought vectors
        max_mse_dist: the maximum euclidean distance between thought vectors
    """
    def __init__(self, embedding_type='paragramcf', max_mse_dist=None, min_cos_sim=None):
        self.word_embedding = WordEmbedding(embedding_type)
        self.embedding_type = embedding_type
        
        if (max_mse_dist or min_cos_sim) is None:
            raise ValueError('Must set max_mse_dist or min_cos_sim')
        
        self.max_mse_dist = max_mse_dist
        self.min_cos_sim = min_cos_sim
    
    @functools.lru_cache(maxsize=2**10)
    def _get_thought_vector(self, tokenized_text):
        """ Sums the embeddings of all the words in `tokenized_text` into a
            "thought vector".
        """
        embeddings = []
        for word in tokenized_text.words:
            embedding = self.word_embedding[word]
            if embedding is not None: # out-of-vocab words do not have embeddings
                embeddings.append(embedding)
        embeddings = torch.tensor(embeddings)
        return torch.sum(embeddings, dim=0)
    
    def __call__(self, x, x_adv, original_text=None):
        """ Returns true if (x, x_adv) are closer than `self.min_cos_sim`
            and `self.max_mse_dist`. """
        
        if not isinstance(x, TokenizedText):
            raise TypeError('x must be of type TokenizedText')
        if not isinstance(x_adv, TokenizedText):
            raise TypeError('x_adv must be of type TokenizedText')
        
        thought_vector_1 = self._get_thought_vector(x)
        thought_vector_2 = self._get_thought_vector(x_adv)
        
        # Check cosine distance.
        if self.min_cos_sim:
            cos_sim = torch.nn.CosineSimilarity(dim=0)(thought_vector_1, thought_vector_2)
            if cos_sim < self.min_cos_sim:
                return False
        # Check MSE distance.
        if self.max_mse_dist:
            mse_dist = torch.sum((thought_vector_1 - thought_vector_2) ** 2)
            if mse_dist > self.max_mse_dist:
                return False
        return True
        
    def extra_repr_keys(self):
        """Set the extra representation of the constraint using these keys.
        
        To print customized extra information, you should reimplement
        this method in your own constraint. Both single-line and multi-line
        strings are acceptable.
        """ 
        if self.min_cos_sim is None:
            metric = 'max_mse_dist'
        else:
            metric = 'min_cos_sim'
        return ['embedding_type', metric]
