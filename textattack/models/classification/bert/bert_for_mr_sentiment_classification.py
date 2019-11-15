from textattack.models.helpers import BERTForClassification

class BERTForMRSentimentClassification(BERTForClassification):
    """ 
    BERT fine-tuned on the Yelp Sentiment dataset for sentiment classification.
    """
    
    MODEL_PATH_CASED = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/BertClassifier/outputs/mr-cased-2019-11-13-14:46'
    MODEL_PATH_UNCASED = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/BertClassifier/outputs/mr-uncased-2019-11-13-13:53'
    def __init__(self, cased=False):
        if cased:
            path = BERTForMRSentimentClassification.MODEL_PATH_CASED
        else:
            path = BERTForMRSentimentClassification.MODEL_PATH_UNCASED
        super().__init__(path)