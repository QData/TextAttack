from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.metrics import Metric
from textattack.constraints.semantics.sentence_encoders import SentenceEncoder
from textattack.shared.utils import LazyLoader

hub = LazyLoader("tensorflow_hub", globals(), "tensorflow_hub")


class USEMetric(Metric):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using the Universal Sentence
    Encoder."""

    def __init__(self, results, **kwargs):
        self.results = results
        self.use_obj = SentenceEncoder()
        self.original_candidates = []
        self.successful_candidates = []

        if kwargs["large"]:
            tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        else:
            tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

        self._tfhub_url = tfhub_url
        # Lazily load the model
        self.model = hub.load(self._tfhub_url)

    def calculate(self):
        for i, result in enumerate(self.results):
            if isinstance(result, FailedAttackResult):
                continue
            elif isinstance(result, SkippedAttackResult):
                continue
            else:
                self.original_candidates.append(
                    result.original_result.attacked_text
                )
                self.successful_candidates.append(
                    result.perturbed_result.attacked_text
                )

        self.use_obj.model = self.model
        self.use_obj.encode = self.encode
        print(self.original_candidates)
        print(self.successful_candidates)
        use_scores = []
        for c in range(len(self.original_candidates)):
            use_scores.append(self.use_obj._sim_score(self.original_candidates[c],self.successful_candidates[c]))

        print(use_scores)

    def encode(self, sentences):
        return self.model(sentences).numpy()