import numpy as np
import os
import torch

from textattack.constraints import Constraint
from textattack.models import InferSent
from textattack.utils import download_if_needed, get_device

class UniversalSentenceEncoder(Constraint):
    """ 
    Constraint using cosine similarity between Universal Sentence Encodings
    of x and x_adv where the text embeddings are created using InferSent.
        
    Args:
        threshold (:obj:`float`, optional): The threshold for the constraint to bet met.
            Defaults to 0.8
        metric (:obj:`str`, optional): The metric function to use. Must be one of TODO. 
            Defaults to cosine. 

    Raises:
        ValueError: If :obj:`metric` is not supported

    """
    
    MODEL_PATH = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/AttackGeneration/infersent-encoder'
    WORD_EMBEDDING_PATH = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/AttackGeneration/word_embeddings'
    
    def __init__(self, threshold=0.8, metric='cosine'):
        self.threshold = threshold
        self.model = self.get_infersent_model()
        self.metric = metric
        
        if self.metric=='cosine':
            self.dist = torch.nn.CosineSimilarity
        else:
            raise ValueError(f'Unsupported metric {metric}.')
    
    def get_infersent_model(self):
        """
        Retrieves the InferSent model. 

        Returns:
            The pretrained InferSent model. 

        """
        infersent_version = 2
        model_path = os.path.join(UniversalSentenceEncoder.MODEL_PATH, f'infersent{infersent_version}.pkl')
        download_if_needed(model_path)
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': infersent_version}
        infersent = InferSent(params_model)
        infersent.load_state_dict(torch.load(model_path))
        W2V_PATH = os.path.join(UniversalSentenceEncoder.WORD_EMBEDDING_PATH, 
            'fastText', 'crawl-300d-2M.vec')
        download_if_needed(W2V_PATH)
        infersent.set_w2v_path(W2V_PATH)
        infersent.build_vocab_k_words(K=100000)
        return infersent
    
    def score(self, x, x_adv):
        """ 
        Returns the metric similarity between embeddings of the text and 
        the perturbed text. 

        Args:
            x (str): The original text
            x_adv (str): The perturbed text

        Returns:
            The similarity between the original and perturbed text using the metric. 

        @TODO should this support multiple sentences for x_adv?

        """
        original_embedding, perturbed_embedding = self.model.encode([x, x_adv], tokenize = True)
        
        original_embedding = torch.tensor(original_embedding).to(get_device())
        perturbed_embedding = torch.tensor(perturbed_embedding).to(get_device())
        
        return self.dist(dim=0)(original_embedding, perturbed_embedding)
    
    def score_list(self, x, x_adv_list):
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
        
        x_text = x.text
        x_adv_list_text = [x_adv.text for x_adv in x_adv_list]
        embeddings = self.model.encode([x_text] + x_adv_list_text, tokenize=True)
        
        original_embedding = torch.tensor(embeddings[0]).to(get_device())
        perturbed_embedding = torch.tensor(embeddings[1:]).to(get_device())
        
        # Repeat original embedding to size of perturbed embedding.
        original_embedding = original_embedding.unsqueeze(dim=0).repeat(len(perturbed_embedding),1)
        
        return self.dist(dim=1)(original_embedding, perturbed_embedding)
    
    def call_many(self, x, x_adv_list, original_text=None):
        """
        Filters the list of perturbed texts so that the similarity between the original text
        and the perturbed text is greater than the :obj:`threshold`. 

        Args:
            x (str): The original text
            x_adv_list (list(str)): A list of perturbed texts
            original_text(:obj:`type`, optional): Defaults to None. 

        Returns:
            A filtered list of perturbed texts where each perturbed text meets the similarity threshold. 

        """
        # @TODO can we rename this function `filter`? (It's a reserved keyword in python)
        scores = self.score_list(x, x_adv_list)
        mask = (scores > self.threshold)
        return np.array(x_adv_list)[mask.cpu().numpy()]
    
    def __call__(self, x, x_adv):
        return self.score(x.text, x_adv.text) >= self.threshold 
