from textattack.models.helpers import BERTForClassification

class BERTForMRSentimentClassification(BERTForClassification):
    """ 
    BERT fine-tuned on the Yelp Sentiment dataset for sentiment classification.
    """
    
    MODEL_PATH = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/BertClassifier/outputs/mr-2019-11-08-16:51/'
    def __init__(self):
        super().__init__(BERTForMRSentimentClassification.MODEL_PATH)