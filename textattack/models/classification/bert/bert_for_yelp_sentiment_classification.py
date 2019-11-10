from textattack.models.helpers import BERTForClassification

class BERTForYelpSentimentClassification(BERTForClassification):
    """ 
    BERT fine-tuned on the Yelp Sentiment dataset for sentiment classification.
    """
    
    MODEL_PATH = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/BertClassifier/outputs/yelp-2019-11-08-18:16'
    def __init__(self):
        super().__init__(BERTForYelpSentimentClassification.MODEL_PATH)