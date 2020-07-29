import torch

import textattack
from textattack.shared import utils
from textattack.shared.validators import validate_model_gradient_word_swap_compatibility
from textattack.transformations import Transformation


class WordSwapGradientBased(Transformation):
    """Uses the model's gradient to suggest replacements for a given word.

    Based off of HotFlip: White-Box Adversarial Examples for Text
    Classification (Ebrahimi et al., 2018).
    https://arxiv.org/pdf/1712.06751.pdf

    Arguments:
        model (nn.Module): The model to attack. Model must have a
            `word_embeddings` matrix and `convert_id_to_word` function.
        top_n (int): the number of top words to return at each index
    """

    def __init__(self, model_wrapper, top_n=1):
        # Unwrap model wrappers. Need raw model for gradient.
        if not isinstance(model_wrapper, textattack.models.wrappers.ModelWrapper):
            raise TypeError(f"Got invalid model wrapper type {type(model_wrapper)}")
        self.model = model_wrapper.model
        self.model_wrapper = model_wrapper
        self.tokenizer = self.model_wrapper.tokenizer
        # Make sure we know how to compute the gradient for this model.
        validate_model_gradient_word_swap_compatibility(self.model)
        # Make sure this model has all of the required properties.
        if not hasattr(self.model, "word_embeddings"):
            raise ValueError(
                "Model needs word embedding matrix for gradient-based word swap"
            )
        if not hasattr(self.model, "lookup_table"):
            raise ValueError("Model needs lookup table for gradient-based word swap")
        if not hasattr(self.model, "zero_grad"):
            raise ValueError("Model needs `zero_grad()` for gradient-based word swap")
        if not hasattr(self.tokenizer, "convert_id_to_word"):
            raise ValueError(
                "Tokenizer needs `convert_id_to_word()` for gradient-based word swap"
            )
        if not hasattr(self.tokenizer, "pad_id"):
            raise ValueError("Tokenizer needs `pad_id` for gradient-based word swap")
        if not hasattr(self.tokenizer, "oov_id"):
            raise ValueError("Tokenizer needs `oov_id` for gradient-based word swap")
        self.loss = torch.nn.CrossEntropyLoss()
        self.pad_id = self.model_wrapper.tokenizer.pad_id
        self.oov_id = self.model_wrapper.tokenizer.oov_id
        self.top_n = top_n
        self.is_black_box = False

    def _get_replacement_words_by_grad(self, attacked_text, indices_to_replace):
        """Returns returns a list containing all possible words to replace
        `word` with, based off of the model's gradient.

        Arguments:
            attacked_text (AttackedText): The full text input to perturb
            word_index (int): index of the word to replace
        """
        self.model.train()
        self.model.emb_layer.embedding.weight.requires_grad = True

        lookup_table = self.model.lookup_table.to(utils.device)
        lookup_table_transpose = lookup_table.transpose(0, 1)

        # get word IDs
        text_ids = self.tokenizer.encode(attacked_text.tokenizer_input)

        # set backward hook on the word embeddings for input x
        emb_hook = Hook(self.model.word_embeddings, backward=True)

        self.model.zero_grad()
        predictions = self._call_model(text_ids)
        original_label = predictions.argmax()
        y_true = torch.Tensor([original_label]).long().to(utils.device)
        loss = self.loss(predictions, y_true)
        loss.backward()

        # grad w.r.t to word embeddings
        emb_grad = emb_hook.output[0].to(utils.device).squeeze()

        # grad differences between all flips and original word (eq. 1 from paper)
        vocab_size = lookup_table.size(0)
        diffs = torch.zeros(len(indices_to_replace), vocab_size)
        indices_to_replace = list(indices_to_replace)
        for j, word_idx in enumerate(indices_to_replace):
            # Make sure the word is in bounds.
            if word_idx >= len(emb_grad):
                continue
            # Get the grad w.r.t the one-hot index of the word.
            b_grads = (
                emb_grad[word_idx].view(1, -1).mm(lookup_table_transpose).squeeze()
            )
            a_grad = b_grads[text_ids[word_idx]]
            diffs[j] = b_grads - a_grad

        # Don't change to the pad token.
        diffs[:, self.tokenizer.pad_id] = float("-inf")

        # Find best indices within 2-d tensor by flattening.
        word_idxs_sorted_by_grad = (-diffs).flatten().argsort()

        candidates = []
        num_words_in_text, num_words_in_vocab = diffs.shape
        for idx in word_idxs_sorted_by_grad.tolist():
            idx_in_diffs = idx // num_words_in_vocab
            idx_in_vocab = idx % (num_words_in_vocab)
            idx_in_sentence = indices_to_replace[idx_in_diffs]
            word = self.tokenizer.convert_id_to_word(idx_in_vocab)
            if (not utils.has_letter(word)) or (len(utils.words_from_text(word)) != 1):
                # Do not consider words that are solely letters or punctuation.
                continue
            candidates.append((word, idx_in_sentence))
            if len(candidates) == self.top_n:
                break

        self.model.eval()
        self.model.emb_layer.embedding.weight.requires_grad = (
            self.model.emb_layer_trainable
        )
        return candidates

    def _call_model(self, text_ids):
        """A helper function to query `self.model` with AttackedText `text`."""
        model_input = torch.tensor([text_ids]).to(textattack.shared.utils.device)
        return self.model(model_input)

    def _get_transformations(self, attacked_text, indices_to_replace):
        """Returns a list of all possible transformations for `text`.

        If indices_to_replace is set, only replaces words at those
        indices.
        """
        transformations = []
        for word, idx in self._get_replacement_words_by_grad(
            attacked_text, indices_to_replace
        ):
            transformations.append(attacked_text.replace_word_at_index(idx, word))
        return transformations

    def extra_repr_keys(self):
        return ["top_n"]


class Hook:
    def __init__(self, module, backward=False):
        if backward:
            self.hook = module.register_backward_hook(self.hook_fn)
        else:
            self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = [x.to(utils.device) for x in input]
        self.output = [x.to(utils.device) for x in output]

    def close(self):
        self.hook.remove()
