from collections import deque
import itertools

import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizerFast

from textattack.shared import utils
from textattack.transformations.word_swap import WordSwap


class WordSwapBERTMaskedLM(WordSwap):
    """
    Generate potential replacements for a word using BERT-Masked LM.

    Based off of two papers
        - "BAE: BERT-based Adversarial Examples for Text Classification" (Garg et al., 2020) https://arxiv.org/abs/2004.01970
        - "BERT-ATTACK: Adversarial Attack Against BERT Using BERT" (Li et al, 2020) https://arxiv.org/abs/2004.09984

    Choose which method to use by specifying "bae" or "bert-attack" for `method_type` argument
    """

    def __init__(
        self,
        method_type="bert-attack",
        max_candidates=10,
        language_model="bert-base-uncased",
        max_length=256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.method_type = method_type
        self.max_candidates = max_candidates
        self.lm_type = language_model
        self.max_length = max_length
        self.subword_expand_limit = 2

        self._lm_tokenizer = BertTokenizerFast.from_pretrained(
            "bert-large-uncased-whole-word-masking"
        )
        self._language_model = BertForMaskedLM.from_pretrained(
            "bert-large-uncased-whole-word-masking"
        )
        self._language_model.to(utils.device)
        self._segment_tensor = (
            torch.zeros(self.max_length, dtype=torch.long).unsqueeze(0).to(utils.device)
        )
        self._language_model.eval()

    def _top_k_bert_pred(self, ids, masked_index, k):
        """
        Query BERTMaskedLM with ids to get top-k predictions for masked position
        """
        ids_tensor = torch.tensor([ids]).to(utils.device)
        with torch.no_grad():
            preds = self._language_model(
                ids_tensor, token_type_ids=self._segment_tensor
            )[0]

        mask_token_logits = preds[0, masked_index]
        topk = torch.topk(mask_token_logits, self.max_candidates)
        top_logits = topk[0].tolist()
        top_ids = topk[1].tolist()

        return top_ids, top_logits

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

        top_ids, _ = self._top_k_bert_pred(ids, masked_index, self.max_candidates)
        replacement_words = []
        for id in top_ids:
            token = self._lm_tokenizer.convert_ids_to_tokens(id)
            if check_if_word(token):
                replacement_words.append(token)

        return replacement_words

    def _bert_attack_replacement_words(self, current_text, index):
        tokenized_text = self._lm_tokenizer.encode_plus(
            current_text.text, max_length=self.max_length, pad_to_max_length=True,
            return_offsets_mapping=True
        )
    
        current_ids = tokenized_text["input_ids"]
        token2char_offset = tokenized_text["offset_mapping"]
        target_word = current_text.words[index]
        # Start and end position of word in the whole text
        word_start, word_end = current_text.words2char_offset[target_word]
        # We need to find which tokens belong to the word we want to replace
        target_ids_pos = []

        for i in range(len(current_ids)):
            id = current_ids[i]
            token_start, token_end = token2char_offset[i]
            token = self._lm_tokenizer.convert_ids_to_tokens(id).replace("##", "")
            if (token_start >= word_start and token_end <= token_end) and token in target_word:
                target_ids_pos.append(i)

        if not target_ids_pos:
            return []
        elif len(target_ids_pos) == 1:
            # Word to replace is tokenized as a single word
            masked_pos = target_ids_pos[0]
            current_ids[masked_pos] = self._lm_tokenizer.mask_token_id
            top_ids, _ = self._top_k_bert_pred(
                current_ids, masked_pos, self.max_candidates
            )
            replacement_words = []
            for id in top_ids:
                token = self._lm_tokenizer.convert_ids_to_tokens(id)
                if check_if_word(token):
                    replacement_words.append(token)
            return replacement_words
        else:
            # Word to replace is tokenized as multiple sub-words
            top_replacements = []
            for i in target_ids_pos:
                ids = current_ids.copy()
                ids[i] = self._lm_tokenizer.mask_token_id
                # `top_results` is tuple of (ids, logits)
                top_ids, top_logits = self._top_k_bert_pred(
                    ids, i, self.max_candidates
                )
                top_replacements.append(list(zip(top_ids, top_logits)))

            products = itertools.product(*top_replacements)
            combination_results = []
            for product in products:
                word = []
                prob = 1
                for sub_word in product:
                    id, p = sub_word
                    word.append(i)
                    prob *= p
                word = "".join(self._lm_tokenizer.convert_ids_to_tokens(word)).replace(
                    "##", ""
                )
                if check_if_word(word):
                    combination_results.append((word, prob))
            # Sort to get top-K results
            sorted(combination_results, key=lambda x: x[1], reverse=True)
            return combination_results[: self.max_candidates]

    def _get_replacement_words(self, current_text, index):
        if self.method_type == "bae":
            return self._bae_replacement_words(current_text, index)
        elif self.method_type == "bert-attack":
            return self._bert_attack_replacement_words(current_text, index)
        else:
            raise ValueError(
                f"Unrecognized value {self.method_type} for `self.method_type`."
            )

    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []

        for i in indices_to_modify:
            replacement_words = self._get_replacement_words(current_text, i)

            print(replacement_words)
            transformed_texts_idx = []
            for r in replacement_words:
                transformed_texts_idx.append(current_text.replace_word_at_index(i, r))
            transformed_texts.extend(transformed_texts_idx)

        return transformed_texts

    def extra_repr_keys(self):
        return ["max_candidates", "lm_type", "max_length"]


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
