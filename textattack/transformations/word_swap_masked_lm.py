import itertools

import numpy as np
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer

from textattack.shared import utils
from textattack.transformations.word_swap import WordSwap


class WordSwapMaskedLM(WordSwap):
    """
    Generate potential replacements for a word using BERT-Masked LM.

    Based off of two papers
        - "BAE: BERT-based Adversarial Examples for Text Classification" (Garg et al., 2020) https://arxiv.org/abs/2004.01970
        - "BERT-ATTACK: Adversarial Attack Against BERT Using BERT" (Li et al, 2020) https://arxiv.org/abs/2004.09984

    Choose which method to use by specifying "bae" or "bert-attack" for `method` argument
    """

    def __init__(
        self,
        method="bae",
        masked_language_model="bert-base-uncased",
        max_length=256,
        max_candidates=50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.method = method
        self.masked_lm_name = masked_language_model
        self.max_length = max_length
        self.max_candidates = max_candidates

        self._lm_tokenizer = AutoTokenizer.from_pretrained(
            masked_language_model, use_fast=True
        )
        self._language_model = AutoModelWithLMHead.from_pretrained(
            masked_language_model
        )
        self._language_model.to(utils.device)
        self._segment_tensor = (
            torch.zeros(self.max_length, dtype=torch.long).unsqueeze(0).to(utils.device)
        )
        self._language_model.eval()

    def _bae_replacement_words(self, current_text, index):
        # TODO: is it necessary to create a new AttackedText to recover case and punctuation?
        masked_attacked_text = current_text.replace_word_at_index(
            index, self._lm_tokenizer.mask_token
        )
        ids = self._lm_tokenizer.encode(
            masked_attacked_text.text,
            max_length=self.max_length,
            pad_to_max_length=True,
        )

        try:
            masked_index = ids.index(self._lm_tokenizer.mask_token_id)
        except ValueError:
            return []

        ids_tensor = torch.tensor([ids]).to(utils.device)
        with torch.no_grad():
            preds = self._language_model(
                ids_tensor, token_type_ids=self._segment_tensor
            )[0]

        mask_token_probs = preds[0, masked_index]
        topk = torch.topk(mask_token_probs, self.max_candidates)
        top_logits = topk[0].tolist()
        top_ids = topk[1].tolist()

        replacement_words = []
        for id in top_ids:
            token = self._lm_tokenizer.convert_ids_to_tokens(id)
            if check_if_word(token):
                replacement_words.append(token)

        return replacement_words

    def _bert_attack_replacement_words(self, current_text, index, extra_args):
        top_pred_ids = extra_args["top_pred_ids"]
        masked_lm_logits = extra_args["masked_lm_logits"]
        current_ids = extra_args["current_ids"]
        token2char_offset = extra_args["token2char_offset"]

        target_word = current_text.words[index]
        # Start and end position of word we want to replace in the whole text
        word_start, word_end = current_text.words2char_offset[target_word]

        # We need to find which BPE tokens belong to the word we want to replace
        # List of indices of tokens that are part of the target word
        target_ids_pos = []

        for i in range(len(current_ids)):
            token_start, token_end = token2char_offset[i]
            token = self._lm_tokenizer.convert_ids_to_tokens(current_ids[i]).replace(
                "##", ""
            )
            if (
                token_start >= word_start and token_end - 1 <= word_end
            ) and token in target_word:
                target_ids_pos.append(i)

        if not target_ids_pos:
            return []
        elif len(target_ids_pos) == 1:
            # Word to replace is tokenized as a single word
            top_preds = top_pred_ids[target_ids_pos[0]].tolist()
            replacement_words = []
            for id in top_preds:
                token = self._lm_tokenizer.convert_ids_to_tokens(id)
                if check_if_word(token):
                    replacement_words.append(token)
            return replacement_words
        else:
            # Word to replace is tokenized as multiple sub-words
            top_preds = [top_pred_ids[i] for i in target_ids_pos]
            products = itertools.product(*top_preds)
            combination_results = []
            # Original BERT-Attack implement uses cross-entropy loss
            cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
            target_ids_pos_tensor = torch.Tensor(target_ids_pos).long()
            word_tensor = torch.zeros(len(target_ids_pos), dtype=torch.long)
            for bpe_tokens in products:
                for i in range(len(bpe_tokens)):
                    word_tensor[i] = bpe_tokens[i]

                logits = torch.index_select(masked_lm_logits, 0, target_ids_pos_tensor)
                loss = cross_entropy_loss(logits, word_tensor)
                perplexity = torch.exp(torch.mean(loss, dim=0)).item()
                word = "".join(
                    self._lm_tokenizer.convert_ids_to_tokens(word_tensor)
                ).replace("##", "")
                if check_if_word(word):
                    combination_results.append((word, perplexity))
            # Sort to get top-K results
            sorted(combination_results, key=lambda x: x[1])
            top_replacements = [
                x[0] for x in combination_results[: self.max_candidates]
            ]
            return top_replacements

    def _get_replacement_words(self, current_text, index, extra_args=None):
        if self.method == "bae":
            return self._bae_replacement_words(current_text, index)
        elif self.method == "bert-attack":
            return self._bert_attack_replacement_words(current_text, index, extra_args)
        else:
            raise ValueError(f"Unrecognized value {self.method} for `self.method`.")

    def _get_transformations(self, current_text, indices_to_modify):
        extra_args = {}
        if self.method == "bert-attack":
            tokenized_text = self._lm_tokenizer.encode_plus(
                current_text.text,
                max_length=self.max_length,
                pad_to_max_length=True,
                return_offsets_mapping=True,
            )

            current_ids = torch.tensor([tokenized_text["input_ids"]]).to(utils.device)
            token2char_offset = tokenized_text["offset_mapping"]
            with torch.no_grad():
                pred_probs = self._language_model(
                    current_ids, token_type_ids=self._segment_tensor
                )[0][0]
            top_probs, top_ids = torch.topk(pred_probs, self.max_candidates)
            extra_args["top_pred_ids"] = top_ids.cpu()
            extra_args["masked_lm_logits"] = pred_probs.cpu()
            extra_args["current_ids"] = tokenized_text["input_ids"]
            extra_args["token2char_offset"] = token2char_offset

        transformed_texts = []

        for i in indices_to_modify:
            replacement_words = self._get_replacement_words(current_text, i, extra_args)
            transformed_texts_idx = []
            for r in replacement_words:
                transformed_texts_idx.append(current_text.replace_word_at_index(i, r))
            transformed_texts.extend(transformed_texts_idx)

        return transformed_texts

    def extra_repr_keys(self):
        return ["method", "masked_lm_name", "max_length", "max_candidates"]


def recover_word_case(word, reference_word):
    """ Makes the case of `word` like the case of `reference_word`. Supports
        lowercase, UPPERCASE, and Capitalized. """
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        # if other, just do not alter the word's case
        return word


def check_if_word(word):
    for c in word:
        if not c.isalpha():
            return False
    return True
