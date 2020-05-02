import os
import tensorflow as tf
import tensorflow_hub as hub

from textattack.constraints.semantics.sentence_encoders import SentenceEncoder
from textattack.shared.utils import get_device

class UniversalSentenceEncoder(SentenceEncoder):
    """ 
    Constraint using similarity between sentence encodings of x and x_adv where 
    the text embeddings are created using the Universal Sentence Encoder.
    """
    def __init__(self, use_version=4, threshold=0.8, large=True, metric='angular', 
            **kwargs):
        if use_version not in [3,4]:
            raise ValueError(f'Unsupported UniversalSentenceEncoder version {use_version}')
        super().__init__(threshold=threshold, metric=metric, **kwargs)
        tfhub_url = 'https://tfhub.dev/google/universal-sentence-encoder{}/{}'.format(
            '-large' if large else '', use_version)
        self.model = hub.load(tfhub_url)
    
    def encode(self, sentences):
        return self.model(sentences)["outputs"].numpy()