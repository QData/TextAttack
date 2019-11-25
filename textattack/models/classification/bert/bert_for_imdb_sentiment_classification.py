from textattack.models.helpers import BERTForClassification

class BERTForIMDBSentimentClassification(BERTForClassification):
    """ 
    BERT fine-tuned on the IMDb Sentiment dataset for sentiment classification.
    """
    
    MODEL_PATH_CASED = '/p/qdata/jm8wx/research/text_attacks/trained_bert_models/imdb-cased'
    MODEL_PATH_UNCASED = '/p/qdata/jm8wx/research/text_attacks/trained_bert_models/imdb-uncased'
    def __init__(self, cased=False):
        if cased:
            path = BERTForIMDBSentimentClassification.MODEL_PATH_CASED
        else:
            path = BERTForIMDBSentimentClassification.MODEL_PATH_UNCASED
        super().__init__(path)
