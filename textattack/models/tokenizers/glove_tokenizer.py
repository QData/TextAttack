"""
Glove Tokenizer
---------------------------------------------------------------------

"""


import json
import tempfile

import tokenizers as hf_tokenizers


class WordLevelTokenizer(hf_tokenizers.implementations.BaseTokenizer):
    """WordLevelTokenizer.

    Represents a simple word level tokenization using the internals of BERT's
    tokenizer.

    Based off the `tokenizers` BertWordPieceTokenizer (https://github.com/huggingface/tokenizers/blob/704cf3fdd2f607ead58a561b892b510b49c301db/bindings/python/tokenizers/implementations/bert_wordpiece.py).
    """

    def __init__(
        self,
        word_id_map={},
        pad_token_id=None,
        unk_token_id=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        lowercase: bool = False,
        unicode_normalizer=None,
    ):
        if pad_token_id:
            word_id_map[pad_token] = pad_token_id
        if unk_token_id:
            word_id_map[unk_token] = unk_token_id
        max_id = max(word_id_map.values())
        for idx, token in enumerate((unk_token, sep_token, cls_token, pad_token)):
            if token not in word_id_map:
                word_id_map[token] = max_id + idx
        # HuggingFace tokenizer expects a path to a `*.json` file to read the
        # vocab from. I think this is kind of a silly constraint, but for now
        # we write the vocab to a temporary file before initialization.
        word_list_file = tempfile.NamedTemporaryFile()
        word_list_file.write(json.dumps(word_id_map).encode())

        word_level = hf_tokenizers.models.WordLevel.from_file(
            word_list_file.name, unk_token=str(unk_token)
        )
        tokenizer = hf_tokenizers.Tokenizer(word_level)

        # Let the tokenizer know about special tokens if they are part of the vocab
        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])
        if tokenizer.token_to_id(str(sep_token)) is not None:
            tokenizer.add_special_tokens([str(sep_token)])
        if tokenizer.token_to_id(str(cls_token)) is not None:
            tokenizer.add_special_tokens([str(cls_token)])
        if tokenizer.token_to_id(str(pad_token)) is not None:
            tokenizer.add_special_tokens([str(pad_token)])

        # Check for Unicode normalization first (before everything else)
        normalizers = []

        if unicode_normalizer:
            normalizers += [
                hf_tokenizers.normalizers.unicode_normalizer_from_str(
                    unicode_normalizer
                )
            ]

        if lowercase:
            normalizers += [hf_tokenizers.normalizers.Lowercase()]

        # Create the normalizer structure
        if len(normalizers) > 0:
            if len(normalizers) > 1:
                tokenizer.normalizer = hf_tokenizers.normalizers.Sequence(normalizers)
            else:
                tokenizer.normalizer = normalizers[0]

        tokenizer.pre_tokenizer = hf_tokenizers.pre_tokenizers.WhitespaceSplit()

        sep_token_id = tokenizer.token_to_id(str(sep_token))
        if sep_token_id is None:
            raise TypeError("sep_token not found in the vocabulary")
        cls_token_id = tokenizer.token_to_id(str(cls_token))
        if cls_token_id is None:
            raise TypeError("cls_token not found in the vocabulary")

        tokenizer.post_processor = hf_tokenizers.processors.BertProcessing(
            (str(sep_token), sep_token_id), (str(cls_token), cls_token_id)
        )

        parameters = {
            "model": "WordLevel",
            "unk_token": unk_token,
            "sep_token": sep_token,
            "cls_token": cls_token,
            "pad_token": pad_token,
            "lowercase": lowercase,
            "unicode_normalizer": unicode_normalizer,
        }

        self.unk_token = unk_token
        self.pad_token = pad_token

        super().__init__(tokenizer, parameters)


class GloveTokenizer(WordLevelTokenizer):
    """A word-level tokenizer with GloVe 200-dimensional vectors.

    Lowercased, since GloVe vectors are lowercased.
    """

    def __init__(
        self, word_id_map={}, pad_token_id=None, unk_token_id=None, max_length=256
    ):
        super().__init__(
            word_id_map=word_id_map,
            unk_token_id=unk_token_id,
            pad_token_id=pad_token_id,
            lowercase=True,
        )
        self.pad_token_id = pad_token_id
        self.oov_token_id = unk_token_id
        self.convert_id_to_word = self.id_to_token
        self.model_max_length = max_length
        # Set defaults.
        self.enable_padding(length=max_length, pad_id=pad_token_id)
        self.enable_truncation(max_length=max_length)

    def _process_text(self, text_input):
        """A text input may be a single-input tuple (text,) or multi-input
        tuple (text, text, ...).

        In the single-input case, unroll the tuple. In the multi-input
        case, raise an error.
        """
        if isinstance(text_input, tuple):
            if len(text_input) > 1:
                raise ValueError(
                    "Cannot use `GloveTokenizer` to encode multiple inputs"
                )
            text_input = text_input[0]
        return text_input

    def encode(self, text):
        text = self._process_text(text)
        return super().encode(text, add_special_tokens=False).ids

    def batch_encode(self, input_text_list):
        """The batch equivalent of ``encode``."""
        input_text_list = list(map(self._process_text, input_text_list))
        encodings = self.encode_batch(
            input_text_list,
            add_special_tokens=False,
        )
        return [x.ids for x in encodings]

    def __call__(self, input_texts):
        if isinstance(input_texts, list):
            return self.batch_encode(input_texts)
        else:
            return self.encode(input_texts)

    def convert_ids_to_tokens(self, ids):
        return [self.convert_id_to_word(_id) for _id in ids]
