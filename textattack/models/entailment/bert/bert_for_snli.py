import torch
from textattack.shared import utils
from textattack.models.helpers import BERTForClassification

class BERTForSNLI(BERTForClassification):
    """ BERT fine-tuned on the SNLI dataset for textual entailment. """
    MODEL_PATH = 'models/entailment/bert/snli-uncased'
    
    def __init__(self):
        super().__init__(BERTForSNLI.MODEL_PATH, num_labels=3)
    
    def __call__(self, *args, **kwargs):
        # Remaps result of normal __call__ to fit the data labels. See below.
        pred = self.model(*args, **kwargs)[0]
        pred = torch.nn.functional.softmax(pred, dim=-1)
        return pred[:, [1,2,0]]

""""
    BERT models are trained with label mapping:
        labeldict = {"contradiction": 0,
                      "entailment": 1,
                      "neutral": 2}
    
    Data label mapping:
        labeldict = {"entailment": 0,
                     "neutral": 1,
                     "contradiction": 2}
    
    Therefore, we need to remap as such:
        0 <- 2
        1 <- 0
        2 <- 1
    
    so that our labels match the true labels of the data.
"""