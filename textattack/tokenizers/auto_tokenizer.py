import transformers
from textattack.tokenizers import Tokenizer
from textattack.shared import TokenizedText

class AutoTokenizer(Tokenizer):
    """ 
    A generic class that convert text to tokens and tokens to IDs. Supports
    any type of tokenization, be it word, wordpiece, or character-based.
    Based on the ``AutoTokenizer`` from the ``transformers`` library.
    
    Args: 
        name: the identifying name of the tokenizer (see AutoTokenizer,
            https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_auto.py)
        max_seq_length: if set, will truncate & pad tokens to fit this length
    """
    def __init__(self, name='bert-base-uncased', max_seq_length=256, pad_to_length=False, use_fast=True):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(name, use_fast=use_fast)
        self.max_seq_length = max_seq_length
    
    def encode(self, input_text):
        encoded_text = self.tokenizer.encode_plus(input_text.split(TokenizedText.SPLIT_TOKEN))
        return dict(encoded_text)