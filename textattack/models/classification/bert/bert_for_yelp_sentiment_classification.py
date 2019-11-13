from textattack.models.helpers import BERTForClassification

class BERTForYelpSentimentClassification(BERTForClassification):
    """ 
    BERT fine-tuned on the Yelp Sentiment dataset for sentiment classification.
    """
    
    # MODEL_PATH = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/BertClassifier/outputs/yelp-2019-11-08-18:16'
    MODEL_PATH_CASED ='/p/qdata/jm8wx/research/text_attacks/RobustNLP/BertClassifier/outputs/yelp-cased-2019-11-11-21:00'
    MODEL_PATH_UNCASED = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/BertClassifier/outputs/yelp-uncased-2019-11-11-11:26'
    def __init__(self):
        if cased:
            path = BERTForYelpSentimentClassification.MODEL_PATH_CASED
        else:
            path = BERTForYelpSentimentClassification.MODEL_PATH_UNCASED
        super().__init__(path)