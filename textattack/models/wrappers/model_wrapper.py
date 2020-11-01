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

    @abstractmethod
    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens."""
        raise NotImplementedError()

    def encode(self, inputs):
        """Helper method that calls ``tokenizer.batch_encode`` if possible, and
        if not, falls back to calling ``tokenizer.encode`` for each input.
        
        Args:
            inputs (list[str]): list of input strings

        Returns:
            tokens (list[list[int]]): List of list of ids
        """"
        if hasattr(self.tokenizer, "batch_encode"):
            return self.tokenizer.batch_encode(inputs)
        else:
            return [self.tokenizer.encode(x) for x in inputs]

    def tokenize(self, inputs, strip=True):
        """Helper method that calls ``tokenizer.tokenize``.
        
        Args:
            inputs (list[str]): list of input strings
            strip (bool): If `True`, we strip auxiliary characters added to tokens (e.g. "##" for BERT, "Ġ" for RoBERTa)
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """"
        tokens = [self.tokenizer.tokenize(x) for x in inputs]
        if strip:
            #`aux_chars` are known auxiliary characters that are added to tokens
            strip_chars = ["##", "Ġ", "__"]
            # Try dummy string "aaaaaaaaaaaaaaaaaaaaaaaaaa" and identify possible prefix
            # TODO: Find a better way to identify prefixes
            strip_charas.append(self.tokenize(["aaaaaaaaaaaaaaaaaaaaaaaaaa"])[0][2].replace("a", ""))
            
            def strip(s, chars):
                for c in chars:
                    s = s.replace(c, "")
                return s

            tokens = [[strip(t, strip_chars) for t in x] for x in tokens]

        return tokens