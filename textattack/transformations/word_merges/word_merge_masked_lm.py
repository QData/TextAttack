"""
WordMergeMaskedLM class
------------------------------------------------

"""
import re

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from textattack.shared import utils
from textattack.transformations.transformation import Transformation


class WordMergeMaskedLM(Transformation):
    """Generate potential merge of adjacent using a masked language model.

    Based off of:
        CLARE: Contextualized Perturbation for Textual Adversarial Attack" (Li et al, 2020) https://arxiv.org/abs/2009.07502

    Args:
        masked_language_model (Union[str|transformers.AutoModelForMaskedLM]): Either the name of pretrained masked language model from `transformers` model hub
            or the actual model. Default is `bert-base-uncased`.
        tokenizer (obj): The tokenizer of the corresponding model. If you passed in name of a pretrained model for `masked_language_model`,
            you can skip this argument as the correct tokenizer can be infered from the name. However, if you're passing the actual model, you must
            provide a tokenizer.
        max_length (int): The max sequence length the masked language model is designed to work with. Default is 512.
        window_size (int): The number of surrounding words to include when making top word prediction.
            For each position to merge, we take `window_size // 2` words to the left and `window_size // 2` words to the right and pass the text within the window
            to the masked language model. Default is `float("inf")`, which is equivalent to using the whole text.
        max_candidates (int): Maximum number of candidates to consider as replacements for each word. Replacements are
            ranked by model's confidence.
        min_confidence (float): Minimum confidence threshold each replacement word must pass.
    """

    def __init__(
        self,
        masked_language_model="bert-base-uncased",
        tokenizer=None,
        max_length=512,
        window_size=float("inf"),
        max_candidates=50,
        min_confidence=5e-4,
        batch_size=16,
    ):
        super().__init__()
        self.max_length = max_length
        self.window_size = window_size
        self.max_candidates = max_candidates
        self.min_confidence = min_confidence
        self.batch_size = batch_size

        if isinstance(masked_language_model, str):
            self._language_model = AutoModelForMaskedLM.from_pretrained(
                masked_language_model
            )
            self._lm_tokenizer = AutoTokenizer.from_pretrained(
                masked_language_model, use_fast=True
            )
        else:
            self._language_model = masked_language_model
            if tokenizer is None:
                raise ValueError(
                    "`tokenizer` argument must be provided when passing an actual model as `masked_language_model`."
                )
            self._lm_tokenizer = tokenizer
        self._language_model.to(utils.device)
        self._language_model.eval()
        self.masked_lm_name = self._language_model.__class__.__name__

    def _encode_text(self, text):
        """Encodes ``text`` using an ``AutoTokenizer``, ``self._lm_tokenizer``.

        Returns a ``dict`` where keys are strings (like 'input_ids') and
        values are ``torch.Tensor``s. Moves tensors to the same device
        as the language model.
        """
        encoding = self._lm_tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.to(utils.device) for k, v in encoding.items()}

    def _get_merged_words(self, current_text, indices_to_modify):
        """Get replacement words for the word we want to replace using BAE
        method.

        Args:
            current_text (AttackedText): Text we want to get replacements for.
            index (int): index of word we want to replace
        """
        masked_texts = []
        for index in indices_to_modify:
            temp_text = current_text.replace_word_at_index(
                index, self._lm_tokenizer.mask_token
            )
            temp_text = temp_text.delete_word_at_index(index + 1)
            # Obtain window
            temp_text = temp_text.text_window_around_index(index, self.window_size)
            masked_texts.append(temp_text)

        i = 0
        # 2-D list where for each index to modify we have a list of replacement words
        replacement_words = []
        while i < len(masked_texts):
            inputs = self._encode_text(masked_texts[i : i + self.batch_size])
            ids = [
                inputs["input_ids"][i].tolist() for i in range(len(inputs["input_ids"]))
            ]
            with torch.no_grad():
                preds = self._language_model(**inputs)[0]

            for j in range(len(ids)):
                try:
                    # Need try-except b/c mask-token located past max_length might be truncated by tokenizer
                    masked_index = ids[j].index(self._lm_tokenizer.mask_token_id)
                except ValueError:
                    replacement_words.append([])
                    continue

                mask_token_logits = preds[j, masked_index]
                mask_token_probs = torch.softmax(mask_token_logits, dim=0)
                ranked_indices = torch.argsort(mask_token_probs, descending=True)
                top_words = []
                for _id in ranked_indices:
                    _id = _id.item()
                    word = self._lm_tokenizer.convert_ids_to_tokens(_id)
                    if utils.check_if_subword(
                        word,
                        self._language_model.config.model_type,
                        (masked_index == 1),
                    ):
                        word = utils.strip_BPE_artifacts(
                            word, self._language_model.config.model_type
                        )
                    if (
                        mask_token_probs[_id] >= self.min_confidence
                        and utils.is_one_word(word)
                        and not utils.check_if_punctuations(word)
                    ):
                        top_words.append(word)

                    if (
                        len(top_words) >= self.max_candidates
                        or mask_token_probs[_id] < self.min_confidence
                    ):
                        break

                replacement_words.append(top_words)

            i += self.batch_size

        return replacement_words

    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []
        indices_to_modify = list(indices_to_modify)
        # find indices that are suitable to merge
        token_tags = [
            current_text.pos_of_word_index(i) for i in range(current_text.num_words)
        ]
        merge_indices = find_merge_index(token_tags)
        merged_words = self._get_merged_words(current_text, merge_indices)
        transformed_texts = []
        for i in range(len(merged_words)):
            index_to_modify = merge_indices[i]
            word_at_index = current_text.words[index_to_modify]
            for word in merged_words[i]:
                word = word.strip("Ġ")
                if word != word_at_index and re.search("[a-zA-Z]", word):
                    temp_text = current_text.delete_word_at_index(index_to_modify + 1)
                    transformed_texts.append(
                        temp_text.replace_word_at_index(index_to_modify, word)
                    )

        return transformed_texts

    def extra_repr_keys(self):
        return ["masked_lm_name", "max_length", "max_candidates", "min_confidence"]


def find_merge_index(token_tags, indices=None):
    merge_indices = []
    if indices is None:
        indices = range(len(token_tags) - 1)
    for i in indices:
        cur_tag = token_tags[i]
        next_tag = token_tags[i + 1]
        if cur_tag == "NOUN" and next_tag == "NOUN":
            merge_indices.append(i)
        elif cur_tag == "ADJ" and next_tag in ["NOUN", "NUM", "ADJ", "ADV"]:
            merge_indices.append(i)
        elif cur_tag == "ADV" and next_tag in ["ADJ", "VERB"]:
            merge_indices.append(i)
        elif cur_tag == "VERB" and next_tag in ["ADV", "VERB", "NOUN", "ADJ"]:
            merge_indices.append(i)
        elif cur_tag == "DET" and next_tag in ["NOUN", "ADJ"]:
            merge_indices.append(i)
        elif cur_tag == "PRON" and next_tag in ["NOUN", "ADJ"]:
            merge_indices.append(i)
        elif cur_tag == "NUM" and next_tag in ["NUM", "NOUN"]:
            merge_indices.append(i)
    return merge_indices
