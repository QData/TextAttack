import torch
from textattack.shared import utils

from textattack.models.helpers import LSTMForClassification


class LSTMForAGNewsClassification(LSTMForClassification):
    """ 
    A long short term memory (LSTM) neural network with reasonable default 
    parameters, trained on the AG News dataset for topic classification. Base 
    embeddings are GLOVE vectors of dimension 200.
    
    Base model in ``textattack.models.helpers.lstm_for_classification``.

    Args:
        max_seq_length(:obj:`int`, optional):  Maximum length of a sequence after tokenizing.
            Defaults to 128.
            
    """
    
    MODEL_PATH = 'models/classification/lstm/ag_news'
    
    def __init__(self, max_seq_length=128):
        super().__init__(max_seq_length=max_seq_length, nclasses=4)
        self.load_from_disk(LSTMForAGNewsClassification.MODEL_PATH)
