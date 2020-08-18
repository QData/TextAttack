from abc import ABC, abstractmethod


class ModelWrapper(ABC):
    """A model wrapper queries a model with a list of text inputs.

    Classification-based models return a list of lists, where each sublist
    represents the model's scores for a given input.

    Text-to-text models return a list of strings, where each string is the
    output – like a translation or summarization – for a given input.
    """

    @abstractmethod
    def __call__(self, text_list):
        raise NotImplementedError()

    def tokenize(self, inputs):
        """Helper method that calls ``tokenizer.batch_encode`` if possible, and
        if not, falls back to calling ``tokenizer.encode`` for each input."""
        if hasattr(self.tokenizer, "batch_encode"):
            return self.tokenizer.batch_encode(inputs)
        else:
            return [self.tokenizer.encode(x) for x in inputs]
