"""
HuggingFace Model Wrapper
--------------------------
"""

import torch
import transformers

import textattack
from textattack.models.helpers import T5ForTextToText
from textattack.models.tokenizers import T5Tokenizer

from .pytorch_model_wrapper import PyTorchModelWrapper

torch.cuda.empty_cache()


class HuggingFaceModelWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, max_length=None):
        assert isinstance(
            model, (transformers.PreTrainedModel, T5ForTextToText)
        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        assert isinstance(
            tokenizer,
            (
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
                T5Tokenizer,
            ),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        # Only used for raw `transformers` encoder-decoder generation models
        # (see the `.generate()` branch in `__call__`). Left unset by default
        # so `.generate()` falls back to the model's own `generation_config`
        # (e.g. many summarization checkpoints ship a sensible `max_length`
        # there) instead of this wrapper silently imposing one.
        #
        # Caveat: leaving this unset does not, by itself, guarantee
        # untruncated output. A checkpoint with no `generation_config`
        # entry for `max_length` still falls back to `transformers`'
        # library-wide default of 20 tokens, same order of magnitude as
        # `T5ForTextToText`'s hardcoded `output_max_length=20`
        # (t5_for_text_to_text.py). Callers doing BLEU/overlap comparisons
        # (`MinimizeBleu`, `NonOverlappingOutput`) against a
        # `ground_truth_output` that's likely to exceed that should pass
        # `max_length` explicitly rather than rely on this default.
        self.max_length = max_length

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            # `T5ForTextToText` (TextAttack's own helper, not a
            # `transformers.PreTrainedModel`) has no `.config` at all, so
            # look it up defensively rather than via `self.model.config`
            # directly.
            model_config = getattr(self.model, "config", None)
            # Prefer `can_generate()` over `hasattr(self.model, "generate")`:
            # on older `transformers` versions, `generate` was defined on
            # every `PreTrainedModel` regardless of whether it actually had
            # a generation-capable head, so `hasattr` alone could misroute a
            # seq2seq-backbone classification model (e.g.
            # `BartForSequenceClassification`, whose config also sets
            # `is_encoder_decoder=True`) into `.generate()`. Fall back to
            # `hasattr` only if this version of `transformers` predates
            # `can_generate()`.
            can_generate = getattr(self.model, "can_generate", None)
            is_generation_model = (
                can_generate()
                if callable(can_generate)
                else hasattr(self.model, "generate")
            )
            if (
                getattr(model_config, "is_encoder_decoder", False)
                and is_generation_model
            ):
                # Seq2seq generation models (e.g. a raw `BartForConditionalGeneration`
                # or `T5ForConditionalGeneration` loaded directly from
                # `transformers`, as opposed to TextAttack's own
                # `T5ForTextToText` helper) need `.generate()` + decoding to
                # produce text. A plain forward pass only returns logits,
                # which breaks text-to-text goal functions (e.g.
                # `NonOverlappingOutput`, `MinimizeBleu`) expecting strings.
                # See https://github.com/QData/TextAttack/issues/771
                generate_kwargs = {
                    "input_ids": inputs_dict["input_ids"],
                    "attention_mask": inputs_dict.get("attention_mask"),
                }
                if self.max_length is not None:
                    generate_kwargs["max_length"] = self.max_length
                generated_ids = self.model.generate(**generate_kwargs)
                return self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
            outputs = self.model(**inputs_dict)

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.

        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]
