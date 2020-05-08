import math
import numpy as np
import os
import torch

from textattack.constraints import Constraint
from textattack.shared import utils

class SentenceEncoder(Constraint):
    """ 
    Constraint using cosine similarity between sentence encodings of x and 
    x_adv.
        
    Args:
        threshold (:obj:`float`, optional): The threshold for the constraint to be met.
            Defaults to 0.8
        metric (:obj:`str`, optional): The similarity metric to use. Defaults to 
            cosine. Options: ['cosine, 'angular']
        compare_with_original (bool): Whether to compare `x_adv` to the previous `x_adv`
            or the original `x`.
        window_size (int): The number of words to use in the similarity 
            comparison.
    """
    
    def __init__(self, threshold=0.8, metric='cosine', compare_with_original=False, window_size=None,
        skip_text_shorter_than_window=False):
        self.metric = metric
        self.threshold = threshold
        self.compare_with_original = compare_with_original
        self.window_size = window_size
        self.skip_text_shorter_than_window = skip_text_shorter_than_window
        
        if metric == 'cosine':
            self.sim_metric = torch.nn.CosineSimilarity(dim=1)
        elif metric == 'angular':
            self.sim_metric = get_angular_sim
        elif metric == 'max_euclidean':
            # If the threshold requires embedding similarity measurement 
            # be less than or equal to a certain value, just negate it,
            # so that we can still compare to the threshold using >=.
            self.threshold = -threshold
            self.sim_metric = get_neg_euclidean_dist
        else:
            raise ValueError(f'Unsupported metric {metric}.')
    
    def encode(self, sentences):
        """ Encodes a list of sentences. To be implemented by subclasses. """
        raise NotImplementedError()
    
    def sim_score(self, x, x_adv):
        """ 
        Returns the metric similarity between embeddings of the text and 
        the perturbed text. 

        Args:
            x (str): The original text
            x_adv (str): The perturbed text

        Returns:
            The similarity between the original and perturbed text using the metric. 

        """
        original_embedding, perturbed_embedding = self.model.encode([x, x_adv])
        
        original_embedding = torch.tensor(original_embedding).to(utils.get_device())
        perturbed_embedding = torch.tensor(perturbed_embedding).to(utils.get_device())
        
        original_embedding = torch.unsqueeze(original_embedding, dim=0)
        perturbed_embedding = torch.unsqueeze(perturbed_embedding, dim=0) 
        
        return self.sim_metric(original_embedding, perturbed_embedding)
    
    def _score_list(self, x, x_adv_list):
        """
        Returns the metric similarity between the embedding of the text and a list
        of perturbed text. 

        Args:
            x (str): The original text
            x_adv_list (list(str)): A list of perturbed texts

        Returns:
            A list with the similarity between the original text and each perturbed text in :obj:`x_adv_list`. 
            If x_adv_list is empty, an empty tensor is returned

        """
        # Return an empty tensor if x_adv_list is empty.
        # This prevents us from calling .repeat(x, 0), which throws an
        # error on machines with multiple GPUs (pytorch 1.2).
        if len(x_adv_list) == 0: return torch.tensor([])
        
        if self.window_size:
            x_list_text = []
            x_adv_list_text = []
            for x_adv in x_adv_list:
                modified_index = x_adv.attack_attrs['modified_word_index']
                x_list_text.append(x.text_window_around_index(modified_index, self.window_size))
                x_adv_list_text.append(x_adv.text_window_around_index(modified_index, self.window_size))
            embeddings = self.encode(x_list_text + x_adv_list_text)
            original_embeddings = torch.tensor(embeddings[:len(x_adv_list)]).to(utils.get_device())
            perturbed_embeddings = torch.tensor(embeddings[len(x_adv_list):]).to(utils.get_device())
        else:
            x_text = x.text
            x_adv_list_text = [x_adv.text for x_adv in x_adv_list]
            embeddings = self.encode([x_text] + x_adv_list_text)
            if isinstance(embeddings[0], torch.Tensor):
                original_embedding = embeddings[0].to(utils.get_device())
            else:
                # If the embedding is not yet a tensor, make it one.
                original_embedding = torch.tensor(embeddings[0]).to(utils.get_device())
                
            if isinstance(embeddings, list):
                # If `encode` did not return a Tensor of all embeddings, combine
                # into a tensor.
                perturbed_embeddings = torch.stack(embeddings[1:]).to(utils.get_device())
            else:
                perturbed_embeddings = torch.tensor(embeddings[1:]).to(utils.get_device())
        
            # Repeat original embedding to size of perturbed embedding.
            original_embeddings = original_embedding.unsqueeze(dim=0).repeat(len(perturbed_embeddings),1)
        
        return self.sim_metric(original_embeddings, perturbed_embeddings)
    
    def call_many(self, x, x_adv_list, original_text=None):
        """
        Filters the list of perturbed texts so that the similarity between the original text
        and the perturbed text is greater than the :obj:`threshold`. 

        Args:
            x (TokenizedText): The original text
            x_adv_list (list(TokenizedText)): A list of perturbed texts
            original_text(:obj:TokenizedText, optional): Defaults to None. 

        Returns:
            A filtered list of perturbed texts where each perturbed text meets the similarity threshold. 

        """
        if self.compare_with_original:
            if original_text:
                scores = self._score_list(original_text, x_adv_list)
            else:
                raise ValueError('Must provide original text when compare_with_original is true.')
        else:
            scores = self._score_list(x, x_adv_list)
        for i, x_adv in enumerate(x_adv_list):
            # Optionally ignore similarity score for sentences shorter than the 
            # window size.
            if self.skip_text_shorter_than_window and len(x_adv.words) < self.window_size: 
                scores[i] = 1
            x_adv.attack_attrs['similarity_score'] = scores[i].item()
        mask = (scores >= self.threshold).cpu().numpy().nonzero()
        return np.array(x_adv_list)[mask]
    
    def __call__(self, x, x_adv):
        return self.sim_score(x.text, x_adv.text) >= self.threshold 

    def extra_repr_keys(self):
        return ['metric', 'threshold', 'compare_with_original', 'window_size', 
            'skip_text_shorter_than_window']

def get_angular_sim(emb1, emb2):
    """ Returns the _angular_ similarity between a batch of vector and a batch 
        of vectors.
    """
    cos_sim = torch.nn.CosineSimilarity(dim=1)(emb1, emb2)
    return 1 - (torch.acos(cos_sim) / math.pi)

def get_neg_euclidean_dist(emb1, emb2):
    """ Returns the Euclidean distance between a batch of vectors and a batch of 
        vectors. 
    """
    return -torch.sum((emb1 - emb2) ** 2, dim=1)