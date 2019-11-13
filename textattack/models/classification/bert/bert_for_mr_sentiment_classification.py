from textattack.models.helpers import BERTForClassification

class BERTForMRSentimentClassification(BERTForClassification):
    """ 
    BERT fine-tuned on the Yelp Sentiment dataset for sentiment classification.
    """
    
    # MODEL_PATH = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/BertClassifier/outputs/mr-2019-11-08-16:51/'
    MODEL_PATH_CASED = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/BertClassifier/outputs/mr-cased-2019-11-11-20:52'
    MODEL_PATH_UNCASED = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/BertClassifier/outputs/mr-uncased-2019-11-11-11:17'
    def __init__(self):
        if cased:
            path = BERTForMRSentimentClassification.MODEL_PATH_CASED
        else:
            path = BERTForMRSentimentClassification.MODEL_PATH_UNCASED
        super().__init__(path)