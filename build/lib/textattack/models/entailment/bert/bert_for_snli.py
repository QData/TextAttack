from textattack.shared import utils
from textattack.models.helpers import BERTForClassification

class BERTForSNLI(BERTForClassification):
    """ BERT fine-tuned on the SNLI dataset for textual entailment. """
    MODEL_PATH = 'models/entailment/bert/snli-uncased'
    
    def __init__(self):
        path = BERTForSNLI.MODEL_PATH
        utils.download_if_needed(path)
        super().__init__(path, entailment=True, num_labels=3)
    
    def __call__(self, *args):
        # Remaps result of normal __call__ to fit the data labels. See below.
        result = super().__call__(*args)
        return result[:, [1,2,0]]


""""
    BERT models are trained with:
        labeldict = {"contradiction": 0,
                      "entailment": 1,
                      "neutral": 2}
    
    non-BERT models are trained with:
        labeldict = {"entailment": 0,
                     "neutral": 1,
                     "contradiction": 2}
    
    Therefore, we need to remap as such:
        0 <- 2
        1 <- 0
        2 <- 1
    
    so that our labels match the true labels of the data.
    """