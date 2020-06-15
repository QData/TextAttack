import torch
from textattack.models.helpers import LSTMForClassification
from textattack.shared import utils

class LSTMForSSTSentimentClassification(LSTMForClassification):
    """ 
    Args:
        max_seq_length(:obj:`int`, optional):  Maximum length of a sequence after tokenizing.
            Defaults to 128.
            
    """
    
    MODEL_PATH = 'models/classification/lstm/sst'
    
    def __init__(self, max_seq_length=128):
        super().__init__(max_seq_length=max_seq_length)
        self.load_from_disk(LSTMForSSTSentimentClassification.MODEL_PATH)
