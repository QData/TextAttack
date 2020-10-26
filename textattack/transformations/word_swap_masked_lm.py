"""
Word Swap by BERT-Masked LM.
============================================
"""


import itertools

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from textattack.shared import utils
from textattack.transformations.word_swap import WordSwap


class WordSwapMaskedLM(WordSwap):
    """Generate potential replacements for a word using BERT-Masked LM.

    Based off of following papers
        - "BAE: BERT-based Adversarial Examples for Text Classification"  https://arxiv.org/abs/2004.01970
        - "BERT-ATTACK: Adversarial Attack Against BERT Using BERT"  https://arxiv.org/abs/2004.09984
        - "Robustness to Modification with Shared Words in Paraphrase Identification https://arxiv.org/abs/1909.02560"
        - "Contextualized Perturbation for Textual Adversarial Attack https://arxiv.org/pdf/2009.07502.pdf"

    BAE simple masks the word we want to replace and selects top-K replacements predicted by the masked language model.

    BERT-Attack instead performs replacement on token level. For words that are consisted of two or more sub-word tokens,
        it takes the top-K replacements for seach sub-word token and produces all possible combinations of the top replacments.
        Then, it selects the top-K combinations based on their perplexity calculated using the masked language model.

    Choose which method to use by specifying "bae" or "bert-attack" for `method` argument.
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
        self._language_model = AutoModelForMaskedLM.from_pretrained(
            masked_language_model
        )
        self._language_model.to(utils.device)
        self._language_model.eval()

    def _encode_text(self, text):
        """Encodes ``text`` using an ``AutoTokenizer``, ``self._lm_tokenizer``.

        Returns a ``dict`` where keys are strings (like 'input_ids') and
        values are ``torch.Tensor``s. Moves tensors to the same device
        as the language model.
        """
        encoding = self._lm_tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.to(utils.device) for k, v in encoding.items()}

    def _bae_replacement_words(self, current_text, index):
        """Get replacement words for the word we want to replace using BAE
        method.

        Args:
            current_text (AttackedText): Text we want to get replacements for.
            index (int): index of word we want to replace
        """
        masked_attacked_text = current_text.replace_word_at_index(
            index, self._lm_tokenizer.mask_token
        )
        inputs = self._encode_text(masked_attacked_text.text)
        ids = inputs["input_ids"].tolist()[0]

        try:
            # Need try-except b/c mask-token located past max_length might be truncated by tokenizer
            masked_index = ids.index(self._lm_tokenizer.mask_token_id)
        except ValueError:
            return []

        with torch.no_grad():
            preds = self._language_model(**inputs)[0]

        mask_token_probs = preds[0, masked_index]
        topk = torch.topk(mask_token_probs, self.max_candidates)
        top_ids = topk[1].tolist()

        replacement_words = []
        for id in top_ids:
            token = self._lm_tokenizer.convert_ids_to_tokens(id)
            if utils.is_one_word(token) and not check_if_subword(token):
                replacement_words.append(token)

        return replacement_words

    def _bert_attack_replacement_words(
        self,
        current_text,
        index,
        id_preds,
        masked_lm_logits,
    ):
        """Get replacement words for the word we want to replace using BERT-
        Attack method.

        Args:
            current_text (AttackedText): Text we want to get replacements for.
            index (int): index of word we want to replace
            id_preds (torch.Tensor): N x K tensor of top-K ids for each token-position predicted by the masked language model.
                N is equivalent to `self.max_length`.
            masked_lm_logits (torch.Tensor): N x V tensor of the raw logits outputted by the masked language model.
                N is equivlaent to `self.max_length` and V is dictionary size of masked language model.
        """
        # We need to find which BPE tokens belong to the word we want to replace
        masked_text = current_text.replace_word_at_index(
            index, self._lm_tokenizer.mask_token
        )
        current_inputs = self._encode_text(masked_text.text)
        current_ids = current_inputs["input_ids"].tolist()[0]
        word_tokens = self._lm_tokenizer.encode(
            current_text.words[index], add_special_tokens=False
        )

        try:
            # Need try-except b/c mask-token located past max_length might be truncated by tokenizer
            masked_index = current_ids.index(self._lm_tokenizer.mask_token_id)
        except ValueError:
            return []

        # List of indices of tokens that are part of the target word
        target_ids_pos = list(
            range(masked_index, min(masked_index + len(word_tokens), self.max_length))
        )

        if not len(target_ids_pos):
            return []
        elif len(target_ids_pos) == 1:
            # Word to replace is tokenized as a single word
            top_preds = id_preds[target_ids_pos[0]].tolist()
            replacement_words = []
            for id in top_preds:
                token = self._lm_tokenizer.convert_ids_to_tokens(id)
                if utils.is_one_word(token) and not check_if_subword(token):
                    replacement_words.append(token)
            return replacement_words
        else:
            # Word to replace is tokenized as multiple sub-words
            top_preds = [id_preds[i] for i in target_ids_pos]
            products = itertools.product(*top_preds)
            combination_results = []
            # Original BERT-Attack implement uses cross-entropy loss
            cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
            target_ids_pos_tensor = torch.tensor(target_ids_pos)
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
                if utils.is_one_word(word):
                    combination_results.append((word, perplexity))
            # Sort to get top-K results
            sorted(combination_results, key=lambda x: x[1])
            top_replacements = [
                x[0] for x in combination_results[: self.max_candidates]
            ]
            return top_replacements

    def _get_replacement_words(self, current_text, index, **kwargs):
        if self.method == "bae":
            return self._bae_replacement_words(current_text, index)
        elif self.method == "bert-attack":
            return self._bert_attack_replacement_words(current_text, index, **kwargs)
        else:
            raise ValueError(f"Unrecognized value {self.method} for `self.method`.")

    def _get_transformations(self, current_text, indices_to_modify):
        # extra_args = {}
        if self.method == "bert-attack":
            current_inputs = self._encode_text(current_text.text)
            with torch.no_grad():
                pred_probs = self._language_model(**current_inputs)[0][0]
            top_probs, top_ids = torch.topk(pred_probs, self.max_candidates)
            id_preds = top_ids.cpu()
            masked_lm_logits = pred_probs.cpu()

        transformed_texts = []

        for i in indices_to_modify:
            word_at_index = current_text.words[i]
            if self.method == "bert-attack":
                replacement_words = self._get_replacement_words(
                    current_text,
                    i,
                    id_preds=id_preds,
                    masked_lm_logits=masked_lm_logits,
                )
            else:
                replacement_words = self._get_replacement_words(current_text, i)
            transformed_texts_idx = []
            for r in replacement_words:
                if r != word_at_index:
                    transformed_texts_idx.append(
                        current_text.replace_word_at_index(i, r)
                    )
            transformed_texts.extend(transformed_texts_idx)

        return transformed_texts

    def extra_repr_keys(self):
        return ["method", "masked_lm_name", "max_length", "max_candidates"]


def recover_word_case(word, reference_word):
    """Makes the case of `word` like the case of `reference_word`.

    Supports lowercase, UPPERCASE, and Capitalized.
    """
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        # if other, just do not alter the word's case
        return word


def check_if_subword(text):
    return True if "##" in text else False
