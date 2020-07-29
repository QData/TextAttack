import textattack

from .model_wrapper import ModelWrapper


class SklearnModelWrapper(ModelWrapper):
    """Loads an sklearn model and tokenizer."""

    def __init__(self, model, tokenizer):
        raise NotImplementedError()

        self.model = model.to(textattack.shared.utils.device)
        self.tokenizer = tokenizer

    def tokenize(self, text_input_list):
        raise NotImplementedError()

    def __call__(self, text_input_list):
        raise NotImplementedError()
