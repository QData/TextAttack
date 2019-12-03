from textattack import utils as utils
from textattack.models.helpers import BERTForClassification

class BERTForMNLI(BERTForClassification):
    """ 
    BERT fine-tuned on the MNLI for textual entailment. 
    """
    MODEL_PATH = '/p/qdata/jm8wx/research_OLD/TextFooler/mnli_model/'
    
    def __init__(self):
        path = BERTForMNLI.MODEL_PATH
        utils.download_if_needed(path)
        super().__init__(path, entailment=True, num_labels=3)
