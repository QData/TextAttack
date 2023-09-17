import logging

import torch
from tqdm import tqdm
import transformers

import textattack

from .coverage import ExtrinsicCoverage

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


class PerplexityCoverage(ExtrinsicCoverage):
    """
    ``PerplexityCoverage`` meausures the average perplexity of a given test datsaet using a language model
    Args:
        language_model(Union[str, torch.nn.Module]): name of the pretrained language model from `transformers`
            or the actual language model as a `torch.nn.Module` class. Default is "gpt2" from `transformers`.
        tokenizer (:obj:``, optional): If `language_model` is not a pretrained model from `transformers, need to provide
            the tokenizer here.
        max_seq_len(:obj:`int`, optional): Max sequence length to consider. If not set and if the language model is a fixed-length model,
            defaults to the max sequence of length of the model.
        batch_size (int): Batch size when calculating perplexity.
    """

    def __init__(
        self, language_model="gpt2", tokenizer=None, max_seq_len=None, stride_size=512
    ):
        if isinstance(language_model, str):
            self.language_model = transformers.AutoModelForCausalLM.from_pretrained(
                language_model
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                language_model, use_fast=True
            )
            self.max_seq_len = (
                max_seq_len if max_seq_len else self.language_model.config.n_positions
            )
            if stride_size > self.max_seq_len:
                raise ValueError(
                    f"Stride size cannot be greater than max sequence length ({stride_size} > {max_seq_len})."
                )
            self.stride_size = stride_size
        else:
            raise ValueError('`PerplexityCoverage` only currently supports "gpt2"')

        self.language_model.to(textattack.shared.utils.device)
        self.language_model.eval()

    def _gpt2_calc_perplexity(self, text):
        encodings = self.tokenizer(text, return_tensors="pt")
        if self.max_seq_len > 0:
            input_ids = encodings.input_ids[:, : self.max_seq_len]
            attention_mask = encodings.attention_mask[:, : self.max_seq_len]

        input_ids = input_ids.to(textattack.shared.utils.device)
        attention_mask = attention_mask.to(textattack.shared.utils.device)

        lls = []
        for i in range(0, input_ids.size(1), self.stride_size):
            begin_loc = max(i + self.stride_size - self.max_seq_len, 0)
            end_loc = min(i + self.stride_size, input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = input_ids[:, begin_loc:end_loc].to(
                textattack.shared.utils.device
            )
            attention_mask = attention_mask[:, begin_loc:end_loc].to(
                textattack.shared.utils.device
            )
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.language_model(
                    input_ids, attention_mask=attention_mask, labels=target_ids
                )
                log_likelihood = outputs[0] * trg_len

            lls.append(log_likelihood)

        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        return ppl.item()

    def __call__(self, testset):
        """
        Returns average perplexity of `testset`
        Args:
            testset: Iterable of strings
        Returns:
            average perplexity (float)
        """
        ppls = []
        for text in tqdm(testset):
            pp = self._gpt2_calc_perplexity(text)
            ppls.append(pp)
        return sum(ppls) / len(testset), ppls
