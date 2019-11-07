import torch
from textattack.models.helpers import CNNForClassification
import textattack.utils as utils

class CNNForIMDBSentimentClassification(CNNForClassification):
    """ 
    A convolutional neural network with reasonable default parameters, trained 
    on the IMDB Movie Review Sentiment dataset for sentiment classification. 
    Base embeddings are GLOVE vectors of dimension 200.
    
    Base model in `textattack.models.helpers.cnn_for_classification`.

    Args:
        max_seq_length(:obj:`int`, optional):  Maximum length of a sequence after tokenizing.
            Defaults to 128.
            
    """
    
    MODEL_PATH = '/p/qdata/jm8wx/research/text_attacks/textfooler_lstm/outputs/cnn/imdb/model.bin'
    
    def __init__(self, max_seq_length=128):
        super().__init__(max_seq_length=max_seq_length)
        state_dict = torch.load(CNNForIMDBSentimentClassification.MODEL_PATH)
        self.load_state_dict(state_dict)
        self.to(utils.get_device())
        self.eval()