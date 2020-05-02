from sentence_transformers import SentenceTransformer

from textattack.constraints.semantics.sentence_encoders import SentenceEncoder
import textattack.shared.utils as utils

class BERT(SentenceEncoder):
    """ 
    Constraint using similarity between sentence encodings of x and x_adv where 
    the text embeddings are created using BERT, trained on NLI data, and fine-
    tuned on the STS benchmark dataset.
    """
    def __init__(self, threshold=0.7, metric='cosine', **kwargs):
        super().__init__(threshold=threshold, metric=metric, **kwargs)
        self.model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
        self.model.to(utils.get_device())
    
    def encode(self, sentences):
        return self.model.encode(sentences)
