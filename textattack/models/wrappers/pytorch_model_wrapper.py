import torch

import textattack

from .model_wrapper import ModelWrapper


class PyTorchModelWrapper(ModelWrapper):
    """Loads a PyTorch model (`nn.Module`) and tokenizer."""

    def __init__(self, model, tokenizer, batch_size=32):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"PyTorch model must be torch.nn.Module, got type {type(model)}"
            )

        self.model = model.to(textattack.shared.utils.device)
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __call__(self, text_input_list):
        model_device = next(self.model.parameters()).device
        ids = self.tokenize(text_input_list)
        ids = torch.tensor(ids).to(model_device)

        with torch.no_grad():
            outputs = textattack.shared.utils.batch_model_predict(
                self.model, ids, batch_size=self.batch_size
            )

        return outputs
