import os

import textattack
from textattack.constraints.semantics.sentence_encoders import SentenceEncoder


class UniversalSentenceEncoder(SentenceEncoder):
    """ 
    Constraint using similarity between sentence encodings of x and x_adv where 
    the text embeddings are created using the Universal Sentence Encoder.
    """

    def __init__(self, threshold=0.8, large=False, metric="angular", **kwargs):
        # Import `tensorflow_hub`, an optional TextAttack dependency.
        textattack.shared.utils.import_optional("tensorflow")
        textattack.shared.utils.import_optional("tensorflow_hub")
        global tensorflow_hub
        import tensorflow_hub

        super().__init__(threshold=threshold, metric=metric, **kwargs)

        if large:
            tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        else:
            tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

        self.model = tensorflow_hub.load(tfhub_url)

    def encode(self, sentences):
        return self.model(sentences).numpy()
