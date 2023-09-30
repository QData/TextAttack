"""
ModelWrapper class
--------------------------

"""

from abc import ABC, abstractmethod


class ModelWrapper(ABC):
    """A model wrapper queries a model with a list of text inputs.

    Classification-based models return a list of lists, where each
    sublist represents the model's scores for a given input.

    Text-to-text models return a list of strings, where each string is
    the output – like a translation or summarization – for a given
    input.
    """

    @abstractmethod
    def __call__(self, text_input_list, **kwargs):
        raise NotImplementedError()

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens."""
        raise NotImplementedError()

    def _tokenize(self, inputs):
        """Helper method for `tokenize`"""
        raise NotImplementedError()

    def tokenize(self, inputs, strip_prefix=False):
        """Helper method that tokenizes input strings
        Args:
            inputs (list[str]): list of input strings
            strip_prefix (bool): If `True`, we strip auxiliary characters added to tokens as prefixes (e.g. "##" for BERT, "Ġ" for RoBERTa)
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        tokens = self._tokenize(inputs)
        if strip_prefix:
            # `aux_chars` are known auxiliary characters that are added to tokens
            strip_chars = ["##", "Ġ", "__"]
            # TODO: Find a better way to identify prefixes. These depend on the model, so cannot be resolved in ModelWrapper.

            def strip(s, chars):
                for c in chars:
                    s = s.replace(c, "")
                return s

            tokens = [[strip(t, strip_chars) for t in x] for x in tokens]

        return tokens
