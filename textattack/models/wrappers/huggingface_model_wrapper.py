import torch

import textattack

from .pytorch_model_wrapper import PyTorchModelWrapper


class HuggingFaceModelWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

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
            input_dict = {k: [_dict[k] for _dict in inputs] for k in inputs[0]}
            input_dict = {
                k: torch.tensor(v).to(textattack.shared.utils.device)
                for k, v in input_dict.items()
            }
            outputs = self.model(**input_dict)
            return outputs[0]

        with torch.no_grad():
            outputs = textattack.shared.utils.batch_model_predict(
                model_predict, ids, batch_size=self.batch_size
            )

        return outputs
