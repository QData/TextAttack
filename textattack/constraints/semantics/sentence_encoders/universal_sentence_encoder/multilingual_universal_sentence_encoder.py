"""
multilingual universal sentence encoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

from textattack.constraints.semantics.sentence_encoders import SentenceEncoder
from textattack.shared.utils import LazyLoader

hub = LazyLoader("tensorflow_hub", globals(), "tensorflow_hub")
tensorflow_text = LazyLoader(
    "tensorflow_text", globals(), "tensorflow_text"
)  # noqa: F401


class MultilingualUniversalSentenceEncoder(SentenceEncoder):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using the Multilingual Universal
    Sentence Encoder."""

    def __init__(self, threshold=0.8, large=False, metric="angular", **kwargs):
        super().__init__(threshold=threshold, metric=metric, **kwargs)
        if large:
            tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
        else:
            tfhub_url = (
                "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
            )

        # TODO add QA SET. Details at: https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3

        self.model = hub.load(tfhub_url)

    def encode(self, sentences):
        return self.model(sentences).numpy()
