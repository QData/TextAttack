import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from textattack.shared import utils

from .word_insertion import WordInsertion


class WordInsertionMaskedLM(WordInsertion):
    """Generate potential insertion for a word using a masked language model.

    Based off of:
    CLARE: Contextualized Perturbation for Textual Adversarial Attack" (Li et al, 2020):
    https://arxiv.org/abs/2009.07502

    Args:
        masked_language_model (Union[str|transformers.AutoModelForMaskedLM]): Either the name of pretrained masked language model from `transformers` model hub
            or the actual model. Default is `bert-base-uncased`.
        tokenizer (obj): The tokenizer of the corresponding model. If you passed in name of a pretrained model for `masked_language_model`,
            you can skip this argument as the correct tokenizer can be infered from the name. However, if you're passing the actual model, you must
            provide a tokenizer.
        max_length (int): the max sequence length the masked language model is designed to work with. Default is 512.
        max_candidates (int): maximum number of candidates to consider inserting for each position. Replacements are
            ranked by model's confidence.
        min_confidence (float): minimum confidence threshold each new word must pass.
    """

    def __init__(
        self,
        masked_language_model="bert-base-uncased",
        tokenizer=None,
        max_length=512,
        max_candidates=50,
        min_confidence=5e-4,
        batch_size=16,
    ):
        super().__init__()
        self.max_length = max_length
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

    def _get_new_words(self, current_text, indices_to_modify):
        """Get replacement words for the word we want to replace using BAE
        method.

        Args:
            current_text (AttackedText): Text we want to get replacements for.
            indices_to_modify (list[int]): list of word indices where we want to insert
        """
        masked_texts = []
        for index in indices_to_modify:
            masked_texts.append(
                current_text.insert_text_before_word_index(
                    index, self._lm_tokenizer.mask_token
                ).text
            )

        i = 0
        # 2-D list where for each index to modify we have a list of replacement words
        new_words = []
        while i < len(masked_texts):
            inputs = self._encode_text(masked_texts[i : i + self.batch_size])
            ids = inputs["input_ids"].tolist()
            with torch.no_grad():
                preds = self._language_model(**inputs)[0]

            for j in range(len(ids)):
                try:
                    # Need try-except b/c mask-token located past max_length might be truncated by tokenizer
                    masked_index = ids[j].index(self._lm_tokenizer.mask_token_id)
                except ValueError:
                    new_words.append([])
                    continue

                mask_token_logits = preds[j, masked_index]
                mask_token_probs = torch.softmax(mask_token_logits, dim=0)
                ranked_indices = torch.argsort(mask_token_probs, descending=True)
                top_words = []
                for _id in ranked_indices:
                    _id = _id.item()
                    token = self._lm_tokenizer.convert_ids_to_tokens(_id)
                    if utils.check_if_subword(
                        token,
                        self._language_model.config.model_type,
                        (masked_index == 1),
                    ):
                        word = utils.strip_BPE_artifacts(
                            token, self._language_model.config.model_type
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

                new_words.append(top_words)

            i += self.batch_size

        return new_words

    def _get_transformations(self, current_text, indices_to_modify):
        indices_to_modify = list(indices_to_modify)
        new_words = self._get_new_words(current_text, indices_to_modify)
        transformed_texts = []
        for i in range(len(new_words)):
            index_to_modify = indices_to_modify[i]
            word_at_index = current_text.words[index_to_modify]
            for word in new_words[i]:
                if word != word_at_index:
                    transformed_texts.append(
                        current_text.insert_text_before_word_index(
                            index_to_modify, word
                        )
                    )
        return transformed_texts

    def extra_repr_keys(self):
        return ["masked_lm_name", "max_length", "max_candidates", "min_confidence"]
