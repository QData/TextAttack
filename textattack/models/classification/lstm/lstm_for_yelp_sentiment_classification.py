import torch
from textattack.models.helpers import LSTMForClassification
import textattack.utils as utils

class LSTMForYelpSentimentClassification(LSTMForClassification):
    
    MODEL_PATH = '/p/qdata/jm8wx/research/text_attacks/textfooler_lstm/outputs/lstm/yelp_polarity/model.bin'
    
    def __init__(self, max_seq_length=128):
        super().__init__(max_seq_length=max_seq_length)
        state_dict = torch.load(LSTMForYelpSentimentClassification.MODEL_PATH)
        self.load_state_dict(state_dict)
        self.to(utils.get_device())
        self.eval()