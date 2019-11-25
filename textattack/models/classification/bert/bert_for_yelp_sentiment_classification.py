from textattack.models.helpers import BERTForClassification

class BERTForYelpSentimentClassification(BERTForClassification):
    """ 
    BERT fine-tuned on the Yelp Sentiment dataset for sentiment classification.
    """
    
    MODEL_PATH_CASED ='/p/qdata/jm8wx/research/text_attacks/trained_bert_models/yelp-cased'
    MODEL_PATH_UNCASED = '/p/qdata/jm8wx/research/text_attacks/trained_bert_models/yelp-uncased'
    def __init__(self, cased=False):
        if cased:
            path = BERTForYelpSentimentClassification.MODEL_PATH_CASED
        else:
            path = BERTForYelpSentimentClassification.MODEL_PATH_UNCASED
        super().__init__(path)
