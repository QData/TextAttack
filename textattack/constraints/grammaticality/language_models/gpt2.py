import os

import torch

from textattack.shared import utils

from .language_model_constraint import LanguageModelConstraint

# temporarily silence W&B to ignore log-in warning
os.environ["WANDB_SILENT"] = "1"


class GPT2(LanguageModelConstraint):
    """A constraint based on the GPT-2 language model.

    from "Better Language Models and Their Implications"
    (openai.com/blog/better-language-models/)
    """

    def __init__(self, **kwargs):
        import transformers

        # re-enable notifications
        os.environ["WANDB_SILENT"] = "0"
        self.model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.to(utils.device)
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        super().__init__(**kwargs)

    def get_log_probs_at_index(self, text_list, word_index):
        """Gets the probability of the word at index `word_index` according to
        GPT-2.

        Assumes that all items in `text_list` have the same prefix up
        until `word_index`.
        """
        prefix = text_list[0].text_until_word_index(word_index)

        if not utils.has_letter(prefix):
            # This language model perplexity is not defined with respect to
            # a word without a prefix. If the prefix is null, just return the
            # log-probability 0.0.
            return torch.zeros(len(text_list), dtype=torch.float)

        token_ids = self.tokenizer.encode(prefix)
        tokens_tensor = torch.tensor([token_ids])
        tokens_tensor = tokens_tensor.to(utils.device)

        with torch.no_grad():
            outputs = self.model(tokens_tensor)
        predictions = outputs[0]

        probs = []
        for attacked_text in text_list:
            next_word_ids = self.tokenizer.encode(attacked_text.words[word_index])
            next_word_prob = predictions[0, -1, next_word_ids[0]]
            probs.append(next_word_prob)

        return probs
