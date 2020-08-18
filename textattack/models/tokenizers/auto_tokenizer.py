import transformers


class AutoTokenizer:
    """A generic class that convert text to tokens and tokens to IDs. Supports
    any type of tokenization, be it word, wordpiece, or character-based. Based
    on the ``AutoTokenizer`` from the ``transformers`` library, but
    standardizes the functionality for TextAttack.

    Args:
        name: the identifying name of the tokenizer, for example, ``bert-base-uncased``
            (see AutoTokenizer,
            https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_auto.py)
        max_length: if set, will truncate & pad tokens to fit this length
    """

    def __init__(
        self, tokenizer_path=None, tokenizer=None, max_length=256, use_fast=True,
    ):
        if not (tokenizer_path or tokenizer):
            raise ValueError("Must pass tokenizer path or tokenizer")
        if tokenizer_path and tokenizer:
            raise ValueError("Cannot pass both tokenizer path and tokenizer")

        if tokenizer_path:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer_path, use_fast=use_fast
            )
        else:
            self.tokenizer = tokenizer
        self.max_length = max_length
        self.save_pretrained = self.tokenizer.save_pretrained

    def encode(self, input_text):
        """Encodes ``input_text``.

        ``input_text`` may be a string or a tuple of strings, depending
        if the model takes 1 or multiple inputs. The
        ``transformers.AutoTokenizer`` will automatically handle either
        case.
        """
        if isinstance(input_text, str):
            input_text = (input_text,)
        encoded_text = self.tokenizer.encode_plus(
            *input_text,
            max_length=self.max_length,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
        )
        return dict(encoded_text)

    def batch_encode(self, input_text_list):
        """The batch equivalent of ``encode``."""
        if hasattr(self.tokenizer, "batch_encode_plus"):
            if isinstance(input_text_list[0], tuple) and len(input_text_list[0]) == 1:
                # Unroll tuples of length 1.
                input_text_list = [t[0] for t in input_text_list]
            encodings = self.tokenizer.batch_encode_plus(
                input_text_list,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
                padding="max_length",
            )
            # Encodings is a `transformers.utils.BatchEncode` object, which
            # is basically a big dictionary that contains a key for all input
            # IDs, a key for all attention masks, etc.
            dict_of_lists = {k: list(v) for k, v in encodings.data.items()}
            list_of_dicts = [
                {key: value[index] for key, value in dict_of_lists.items()}
                for index in range(max(map(len, dict_of_lists.values())))
            ]
            # We need to turn this dict of lists into a dict of lists.
            return list_of_dicts
        else:
            return [self.encode(input_text) for input_text in input_text_list]
