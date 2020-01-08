from textattack import utils as utils
from textattack.models.helpers import BERTForClassification

class BERTForAGNewsClassification(BERTForClassification):
    """ BERT fine-tuned on the AG News dataset for news topic 
        classification. 
    """
    MODEL_PATH_CASED = 'models/classification/bert/ag-news-cased'
    MODEL_PATH_UNCASED = 'models/classification/bert/ag-news-uncased'
    def __init__(self, cased=False):
        if cased:
            path = BERTForAGNewsClassification.MODEL_PATH_CASED
        else:
            path = BERTForAGNewsClassification.MODEL_PATH_UNCASED
        utils.download_if_needed(path)
        super().__init__(path, num_labels=4)
