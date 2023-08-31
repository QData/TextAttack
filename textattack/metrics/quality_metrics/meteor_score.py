"""

MeteorMetric class:
-------------------------------------------------------
Class for calculating METEOR score on AttackResults

"""

import nltk

from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.metrics import Metric


class MeteorMetric(Metric):
    def __init__(self, **kwargs):
        self.original_candidates = []
        self.successful_candidates = []
        self.all_metrics = {}

    def calculate(self, results):
        """Calculates average Metero score on all successfull attacks.

        Args:
            results (``AttackResult`` objects):
                Attack results for each instance in dataset

        Example::


            >> import textattack
            >> import transformers
            >> model = transformers.AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            >> tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            >> model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
            >> attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)
            >> dataset = textattack.datasets.HuggingFaceDataset("glue", "sst2", split="train")
            >> attack_args = textattack.AttackArgs(
                num_examples=1,
                log_to_csv="log.csv",
                checkpoint_interval=5,
                checkpoint_dir="checkpoints",
                disable_stdout=True
            )
            >> attacker = textattack.Attacker(attack, dataset, attack_args)
            >> results = attacker.attack_dataset()
            >> sbertm = textattack.metrics.quality_metrics.MeteorMetric().calculate(results)
        """

        self.results = results

        for i, result in enumerate(self.results):
            if isinstance(result, FailedAttackResult):
                continue
            elif isinstance(result, SkippedAttackResult):
                continue
            else:
                self.original_candidates.append(
                    result.original_result.attacked_text.text
                )
                self.successful_candidates.append(
                    result.perturbed_result.attacked_text.text
                )

        meteor_scores = []
        for c in range(len(self.original_candidates)):
            meteor_scores.append(
                nltk.translate.meteor(
                    [nltk.word_tokenize(self.original_candidates[c])],
                    nltk.word_tokenize(self.successful_candidates[c]),
                )
            )

        self.all_metrics["avg_attack_meteor_score"] = round(
            sum(meteor_scores) / len(meteor_scores), 2
        )

        return self.all_metrics
