from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.metrics import Metric
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder



class USEMetric(Metric):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using the Universal Sentence
    Encoder."""

    def __init__(self, results, **kwargs):
        self.results = results
        self.use_obj = UniversalSentenceEncoder()
        self.use_obj.model = UniversalSentenceEncoder()
        self.original_candidates = []
        self.successful_candidates = []
        self.all_metrics = {}


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

        
        use_scores = []
        for c in range(len(self.original_candidates)):
            use_scores.append(self.use_obj._sim_score(self.original_candidates[c],self.successful_candidates[c]).item())

        print(use_scores)

        self.all_metrics['avg_attack_use_score'] = sum(use_scores)/len(use_scores)

        return self.all_metrics