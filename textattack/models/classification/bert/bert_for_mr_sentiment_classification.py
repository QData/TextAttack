from textattack.models.helpers import BERTForClassification

class BERTForMRSentimentClassification(BERTForClassification):
    """ BERT fine-tuned on the MR Sentiment dataset for sentiment 
        classification.
    """
    
    MODEL_PATH_CASED = 'models/classification/bert/mr-cased'
    MODEL_PATH_UNCASED = 'models/classification/bert/mr-uncased'
    def __init__(self, cased=False):
        if cased:
            path = BERTForMRSentimentClassification.MODEL_PATH_CASED
        else:
            path = BERTForMRSentimentClassification.MODEL_PATH_UNCASED
        super().__init__(path)
