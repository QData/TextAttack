import torch
from textattack.models.helpers import CNNForClassification
import textattack.utils as utils

class CNNForYelpSentimentClassification(CNNForClassification):
    
    MODEL_PATH = '/p/qdata/jm8wx/research/text_attacks/textfooler_lstm/outputs/cnn/yelp_polarity/model.bin'
    
    def __init__(self, max_seq_length=128):
        super().__init__(max_seq_length=max_seq_length)
        state_dict = torch.load(CNNForYelpSentimentClassification.MODEL_PATH)
        self.load_state_dict(state_dict)
        self.to(utils.get_device())
        self.eval()