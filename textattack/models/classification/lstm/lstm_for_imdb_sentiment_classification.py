import torch
from textattack.models.helpers import LSTMForClassification
from textattack.shared import utils

class LSTMForIMDBSentimentClassification(LSTMForClassification):
    """ 
    A long short term memory (LSTM) neural network with reasonable default 
    parameters, trained on the IMDB Movie Review Sentiment dataset for sentiment 
    classification. Base embeddings are GLOVE vectors of dimension 200.
    
    Base model in ``textattack.models.helpers.lstm_for_classification``.

    Args:
        max_seq_length(:obj:`int`, optional):  Maximum length of a sequence after tokenizing.
            Defaults to 128.
    """
    
    MODEL_PATH = 'models/classification/lstm/imdb'
    
    def __init__(self, max_seq_length=128):
        super().__init__(max_seq_length=max_seq_length)
        self.load_from_disk(LSTMForIMDBSentimentClassification.MODEL_PATH)
