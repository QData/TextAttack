import re

import torch
import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from textattack.metrics import Metric


class Perplexity(Metric):
    """Calculates average Perplexity on all successfull attacks using a pre-trained small GPT-2 model

    Args:
    results (:obj::`list`:class:`~textattack.goal_function_results.GoalFunctionResult`):
                    Attack results for each instance in dataset
    """

    def __init__(self, results):
        self.results = results
        self.all_metrics = {}
        self.original_candidates = []
        self.successful_candidates = []
        self.ppl_model = GPT2LMHeadModel.from_pretrained("gpt2")
        if torch.cuda.is_available():
            self.ppl_model.cuda()
        self.ppl_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.ppl_model.eval()

        self.original_candidates_ppl = []
        self.successful_candidates_ppl = []

    def calculate(self):

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

        self.all_metrics["avg_original_perplexity"] = round(
            self.calc_ppl(self.original_candidates)[0], 2
        )
        self.all_metrics["avg_attack_perplexity"] = round(
            self.calc_ppl(self.successful_candidates)[0], 2
        )

        return self.all_metrics

    def calc_ppl(self, texts):
        eval_loss = 0
        ppl_losses = []
        nb_eval_steps = 0

        with torch.no_grad():
            for text in texts:
                text = self.process_string(text)
                input_ids = torch.tensor(
                    self.ppl_tokenizer.encode(
                        text, add_special_tokens=True, truncation=True
                    )
                )
                if len(input_ids) < 2:
                    continue
                if torch.cuda.is_available():
                    self.input_ids.cuda()
                outputs = self.ppl_model(input_ids, labels=input_ids)
                lm_loss = outputs[0]
                eval_loss += lm_loss.mean().item()
                ppl_losses.append(torch.exp(torch.tensor(lm_loss.mean().item())))
                nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))

        return perplexity.item(), ppl_losses

    def process_string(self, string):
        string = re.sub("( )('[(m)(d)(t)(ll)(re)(ve)(s)])", r"\2", string)
        string = re.sub("(\d+)( )([,\.])( )(\d+)", r"\1\3\5", string)
        # U . S . -> U.S.
        string = re.sub("(\w)( )(\.)( )(\w)( )(\.)", r"\1\3\5\7", string)
        # reduce left space
        string = re.sub("( )([,\.!?:;)])", r"\2", string)
        # reduce right space
        string = re.sub("([(])( )", r"\1", string)
        string = re.sub("s '", "s'", string)
        # reduce both space
        string = re.sub("(')( )(\S+)( )(')", r"\1\3\5", string)
        string = re.sub('(")( )(\S+)( )(")', r"\1\3\5", string)
        string = re.sub("(\w+) (-+) (\w+)", r"\1\2\3", string)
        string = re.sub("(\w+) (/+) (\w+)", r"\1\2\3", string)
        # string = re.sub(" ' ", "'", string)
        return string
