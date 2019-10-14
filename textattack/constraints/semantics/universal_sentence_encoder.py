from models import InferSent
from utils import download_if_needed

import numpy as np
import torch

from .constraint import SentenceConstraint

class UniversalSentenceEncoder(SentenceConstraint):
    """ Constraint using cosine similarity between Universal Sentence Encodings
        of x and x_adv.
        
        Uses InferSent sentence embeddings. """
    
    MODEL_PATH = '/p/qdata/jm8wx/research/RobustNLP/AttackGeneration/infersent-encoder'
    WORD_EMBEDDING_PATH = '/p/qdata/jm8wx/research/RobustNLP/AttackGeneration/word_embeddings'
    
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.model = self.get_infersent_model()
    
    def get_infersent_model(self):
        infersent_version = 2
        model_path = os.path.join(UniversalSentenceEncoder.MODEL_PATH, f'infersent{infersent_version}.pkl')
        utils.download_if_needed(model_path)
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': infersent_version}
        infersent = InferSent(params_model)
        infersent.load_state_dict(torch.load(MODEL_PATH))
        W2V_PATH = os.path.join(UniversalSentenceEncoder.WORD_EMBEDDING_PATH, 
            'fastText', 'crawl-300d-2M.vec')
        infersent.set_w2v_path(W2V_PATH)
        infersent.build_vocab_k_words(K=100000)
        return infersent
    
    def score(self, x, x_adv):
        """ Returns the cosine similarity between embeddings of text x and 
            x_adv. 
            
            @TODO should this support multiple sentences for x_adv?
        """
        original_embedding, perturbed_embedding = self.model.encode([x, x_adv], tokenize = True)[0]
        
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(original_embedding, perturbed_embedding)
    
    def __call__(self, x, x_adv):
        return self.score(x, x_adv) >= self.threshold