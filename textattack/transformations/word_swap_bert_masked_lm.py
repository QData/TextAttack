import itertools
import math
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

        self._lm_tokenizer = BertTokenizerFast.from_pretrained(masked_language_model)
        self._language_model = BertForMaskedLM.from_pretrained(masked_language_model)
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
        P = extra_args['P_matrix']
        current_ids = extra_args['current_ids']
        token2char_offset = extra_args['token2char_offset']

        target_word = current_text.words[index]
        # Start and end position of word we want to replace in the whole text
        word_start, word_end = current_text.words2char_offset[target_word]

        # We need to find which BPE tokens belong to the word we want to replace
        # List of indices of tokens that are part of the target word 
        target_ids_pos = []

        for i in range(len(current_ids)):
            token_start, token_end = token2char_offset[i]
            token = self._lm_tokenizer.convert_ids_to_tokens(current_ids[i]).replace("##", "")
            if (
                token_start >= word_start and token_end-1 <= word_end
            ) and token in target_word:
                target_ids_pos.append(i)

        if not target_ids_pos:
            return []
        elif len(target_ids_pos) == 1:
            # Word to replace is tokenized as a single word
            top_preds = [x[0] for x in P[target_ids_pos[0]]]
            replacement_words = []
            for id in top_preds:
                token = self._lm_tokenizer.convert_ids_to_tokens(id)
                if check_if_word(token):
                    replacement_words.append(token)
            return replacement_words
        else:
            # Word to replace is tokenized as multiple sub-words
            top_preds = [P[i] for i in target_ids_pos]
            products = itertools.product(*top_preds)
            combination_results = []
        
            for product in products:
                word = []
                # Original BERT-Attack implement uses cross-entropy loss
                loss = 0
                for sub_word in product:
                    id, p = sub_word
                    word.append(i)
                    loss += -1 * math.log(p)
                loss = math.exp(loss / len(product))
                word = "".join(self._lm_tokenizer.convert_ids_to_tokens(word)).replace(
                    "##", ""
                )
                if check_if_word(word):
                    combination_results.append((word, prob))
            # Sort to get top-K results
            sorted(combination_results, key=lambda x: x[1], reverse=True)
            return combination_results[: self.max_candidates]

    def _get_replacement_words(self, current_text, index, extra_args=None):
        if self.method == "bae":
            return self._bae_replacement_words(current_text, index)
        elif self.method == "bert-attack":
            return self._bert_attack_replacement_words(current_text, index, extra_args)
        else:
            raise ValueError(
                f"Unrecognized value {self.method} for `self.method`."
            )

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
            dim = top_probs.size()
            # P is `self.max_length` x `self.max_candidates` tensor representing top-k preds for each token.
            # Each element of P is (id, prob) tuple.
            # We would like to use torch.stack, but ids are int while probs are float, so not possible. 
            P_matrix = [[(top_ids[i][j].item(), top_probs[i][j].item()) for j in range(dim[1])] for i in range(dim[0])]
            extra_args['P_matrix'] = P_matrix
            extra_args['current_ids'] = tokenized_text["input_ids"]
            extra_args['token2char_offset'] = token2char_offset

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
