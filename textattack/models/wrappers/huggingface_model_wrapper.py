import torch
import transformers

import textattack

from .pytorch_model_wrapper import PyTorchModelWrapper


class HuggingFaceModelWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, batch_size=32):
        self.model = model.to(textattack.shared.utils.device)
        if isinstance(tokenizer, transformers.PreTrainedTokenizer):
            tokenizer = textattack.models.tokenizers.AutoTokenizer(tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        ids = self.tokenize(text_input_list)

        def model_predict(inputs):
            """Turn a list of dicts into a dict of lists.

            Then make lists (values of dict) into tensors.
            """
            model_device = next(self.model.parameters()).device
            input_dict = {k: [_dict[k] for _dict in inputs] for k in inputs[0]}
            input_dict = {
                k: torch.tensor(v).to(model_device) for k, v in input_dict.items()
            }
            outputs = self.model(**input_dict)

            if isinstance(outputs[0], str):
                # HuggingFace sequence-to-sequence models return a list of
                # string predictions as output. In this case, return the full
                # list of outputs.
                return outputs
            else:
                # HuggingFace classification models return a tuple as output
                # where the first item in the tuple corresponds to the list of
                # scores for each input.
                return outputs[0]

        with torch.no_grad():
            outputs = textattack.shared.utils.batch_model_predict(
                model_predict, ids, batch_size=self.batch_size
            )

        return outputs
