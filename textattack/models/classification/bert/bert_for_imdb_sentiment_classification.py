from textattack.models.helpers import BERTForClassification

class BERTForIMDBSentimentClassification(BERTForClassification):
    """ 
    BERT fine-tuned on the Yelp Sentiment dataset for sentiment classification.
    """
    
    # MODEL_PATH = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/BertClassifier/outputs/imdb-2019-11-08-17:12'
    MODEL_PATH_CASED = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/BertClassifier/outputs/imdb-cased-2019-11-12-05:04'
    MODEL_PATH_UNCASED = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/BertClassifier/outputs/imdb-uncased-2019-11-11-19:57'
    def __init__(self, cased=False):
        if cased:
            path = BERTForIMDBSentimentClassification.MODEL_PATH_CASED
        else:
            path = BERTForIMDBSentimentClassification.MODEL_PATH_UNCASED
        super().__init__(path)