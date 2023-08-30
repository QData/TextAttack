"""

Perplexity Metric:
-------------------------------------------------------
Class for calculating perplexity from AttackResults

"""

import torch

from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.metrics import Metric
import textattack.shared.utils


class Perplexity(Metric):
    def __init__(self, model_name="gpt2"):
        self.all_metrics = {}
        self.original_candidates = []
        self.successful_candidates = []

        if model_name == "gpt2":
            from transformers import GPT2LMHeadModel, GPT2Tokenizer

            self.ppl_model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.ppl_model.to(textattack.shared.utils.device)
            self.ppl_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.ppl_model.eval()
            self.max_length = self.ppl_model.config.n_positions
        else:
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            self.ppl_model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.ppl_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.ppl_model.to(textattack.shared.utils.device)
            self.ppl_model.eval()
            self.max_length = self.ppl_model.config.max_position_embeddings

        self.stride = 512

    def calculate(self, results):
        """Calculates average Perplexity on all successfull attacks using a
        pre-trained small GPT-2 model.

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
            >> ppl = textattack.metrics.quality_metrics.Perplexity().calculate(results)
        """
        self.results = results
        self.original_candidates_ppl = []
        self.successful_candidates_ppl = []

        for i, result in enumerate(self.results):
            if isinstance(result, FailedAttackResult):
                continue
            elif isinstance(result, SkippedAttackResult):
                continue
            else:
                self.original_candidates.append(
                    result.original_result.attacked_text.text.lower()
                )
                self.successful_candidates.append(
                    result.perturbed_result.attacked_text.text.lower()
                )

        ppl_orig = self.calc_ppl(self.original_candidates)
        ppl_attack = self.calc_ppl(self.successful_candidates)

        self.all_metrics["avg_original_perplexity"] = round(ppl_orig, 2)

        self.all_metrics["avg_attack_perplexity"] = round(ppl_attack, 2)

        return self.all_metrics

    def calc_ppl(self, texts):
        with torch.no_grad():
            text = " ".join(texts)
            eval_loss = []
            input_ids = torch.tensor(
                self.ppl_tokenizer.encode(text, add_special_tokens=True)
            ).unsqueeze(0)
            # Strided perplexity calculation from huggingface.co/transformers/perplexity.html
            for i in range(0, input_ids.size(1), self.stride):
                begin_loc = max(i + self.stride - self.max_length, 0)
                end_loc = min(i + self.stride, input_ids.size(1))
                trg_len = end_loc - i
                input_ids_t = input_ids[:, begin_loc:end_loc].to(
                    textattack.shared.utils.device
                )
                target_ids = input_ids_t.clone()
                target_ids[:, :-trg_len] = -100

                outputs = self.ppl_model(input_ids_t, labels=target_ids)
                log_likelihood = outputs[0] * trg_len

                eval_loss.append(log_likelihood)

        return torch.exp(torch.stack(eval_loss).sum() / end_loc).item()
