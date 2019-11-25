import tensorflow as tf
import tensorflow_hub as hub

from textattack.constraints.semantics.sentence_encoders import SentenceEncoder
from textattack.utils import get_device

class UniversalSentenceEncoder(SentenceEncoder):
    """ 
    Constraint using similarity between sentence encodings of x and x_adv where 
    the text embeddings are created using the Universal Sentence Encoder.
    """
    def __init__(self, use_version=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if use_version not in [3,4]:
            raise ValueError(f'Unsupported UniversalSentenceEncoder version {use_version}')
        self.model = hub.load(f'https://tfhub.dev/google/universal-sentence-encoder/{use_version}')
    
    def encode(self, sentences):
        return self.model(sentences)["outputs"].numpy()