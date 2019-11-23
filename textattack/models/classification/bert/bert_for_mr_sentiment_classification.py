from textattack.models.helpers import BERTForClassification

class BERTForMRSentimentClassification(BERTForClassification):
    """ 
    BERT fine-tuned on the MR Sentiment dataset for sentiment classification.
    """
    
    MODEL_PATH_CASED = '/p/qdata/jm8wx/research/text_attacks/trained_bert_models/mr-cased'
    MODEL_PATH_UNCASED = '/p/qdata/jm8wx/research/text_attacks/trained_bert_models/mr-uncased'
    def __init__(self, cased=False):
        if cased:
            path = BERTForMRSentimentClassification.MODEL_PATH_CASED
        else:
            path = BERTForMRSentimentClassification.MODEL_PATH_UNCASED
        super().__init__(path)
        
    def __str__(self):
        return "BERT for MR Sentiment Classification"
