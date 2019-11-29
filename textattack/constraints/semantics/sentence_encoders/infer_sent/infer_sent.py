import numpy as np
import os
import torch

from textattack.constraints import Constraint
from textattack.constraints.semantics.sentence_encoders import SentenceEncoder

import textattack.utils as utils

from .infer_sent_model import InferSentModel

class InferSent(SentenceEncoder):
    """ 
    Constraint using similarity between sentence encodings of x and x_adv where 
    the text embeddings are created using InferSent.
    """
    
    MODEL_PATH = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/AttackGeneration/infersent-encoder'
    WORD_EMBEDDING_PATH = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/AttackGeneration/word_embeddings'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.get_infersent_model()
        self.model.to(utils.get_device())
    
    def get_infersent_model(self):
        """
        Retrieves the InferSent model. 

        Returns:
            The pretrained InferSent model. 

        """
        infersent_version = 2
        model_path = os.path.join(InferSent.MODEL_PATH, f'infersent{infersent_version}.pkl')
        utils.download_if_needed(model_path)
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': infersent_version}
        infersent = InferSentModel(params_model)
        infersent.load_state_dict(torch.load(model_path))
        W2V_PATH = os.path.join(InferSent.WORD_EMBEDDING_PATH, 'fastText', 
            'crawl-300d-2M.vec')
        utils.download_if_needed(W2V_PATH)
        infersent.set_w2v_path(W2V_PATH)
        infersent.build_vocab_k_words(K=100000)
        return infersent
    
    def encode(self, sentences):
        return self.model.encode(sentences, tokenize=True)