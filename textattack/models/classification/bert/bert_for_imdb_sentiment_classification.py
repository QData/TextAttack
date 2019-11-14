from textattack.models.helpers import BERTForClassification

class BERTForIMDBSentimentClassification(BERTForClassification):
    """ 
    BERT fine-tuned on the Yelp Sentiment dataset for sentiment classification.
    """
    
    MODEL_PATH = '/p/qdata/jm8wx/research/text_attacks/RobustNLP/BertClassifier/outputs/imdb-2019-11-08-17:12'
    def __init__(self):
        super().__init__(BERTForIMDBSentimentClassification.MODEL_PATH)
        
    def __str__(self):
        return "BERT for IMDb Sentiment Classification"