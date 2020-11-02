import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from textattack.shared import utils
from textattack.transformations import Transformation


class WordInsertionMaskedLM(Transformation):
    """Generate potential insertion for a word using a masked language model.

    Based off of:
    CLARE: Contextualized Perturbation for Textual Adversarial Attack" (Li et al, 2020):
    https://arxiv.org/abs/2009.07502

    Args:
        masked_language_model (str): the name of pretrained masked language model from `transformers` model hub. Default
         is `bert-base-uncased`.

        max_length (int): the max sequence length the masked language model is designed to work with. Default is 512.

        max_candidates (int): maximum number of candidates to consider as replacements for each word. Replacements are
        ranked by model's confidence.

        min_confidence (float): minimum confidence threshold each replacement word must pass.
    """

    def __init__(
        self,
        masked_language_model="bert-base-uncased",
        max_length=512,
        max_candidates=50,
        min_confidence=5e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.masked_lm_name = masked_language_model
        self.max_length = max_length
        self.max_candidates = max_candidates
        self.min_confidence = min_confidence

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
        masked_attacked_text = current_text.insert_text_after_word_index(
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
            # don't understand this
            preds = self._language_model(**inputs)[0]

        mask_token_logits = preds[0, masked_index]
        mask_token_probs = torch.softmax(mask_token_logits, dim=0)
        ranked_indices = torch.argsort(mask_token_probs)

        replacement_words = []
        for _id in ranked_indices:
            _id = _id.item()
            token = self._lm_tokenizer.convert_ids_to_tokens(_id)
            if utils.is_one_word(token) and not check_if_subword(token):
                if mask_token_probs[_id] > self.min_confidence:
                    replacement_words.append(token)

            if len(replacement_words) >= self.max_candidates:
                break

        return replacement_words

    def _get_replacement_words(self, current_text, index, **kwargs):
        return self._bae_replacement_words(current_text, index)

    def _get_transformations(self, current_text, indices_to_modify):
        # extra_args = {}

        transformed_texts = []

        for i in indices_to_modify:
            word_at_index = current_text.words[i]
            replacement_words = self._get_replacement_words(
                current_text,
                i,
            )

            transformed_texts_idx = []
            for r in replacement_words:
                transformed_texts_idx.append(
                    current_text.insert_text_after_word_index(i, r)
                )
            transformed_texts.extend(transformed_texts_idx)

        return transformed_texts

    def extra_repr_keys(self):
        return ["masked_lm_name", "max_length", "max_candidates"]


def check_if_subword(text):
    return True if "##" in text else False
