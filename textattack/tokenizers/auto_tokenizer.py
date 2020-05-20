import transformers
from textattack.tokenizers import Tokenizer

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
    def __init__(self, name='bert-base-uncased', max_seq_length=None):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        self.max_seq_length = max_seq_length

    def convert_text_to_tokens(self, input_text):
        """ 
        Takes a string input, tokenizes, formats, and returns a list of tokens.
        
        Args:
            input_text (str): The text to tokenize.

        Returns:
            A list of tokens.
        """

        tokens = self.tokenizer.tokenize(input_text)
        return tokens
    
    def convert_tokens_to_ids(self, tokens):
        """ 
        Takes a list of tokens and returns a tensor with text IDs. 
        
        Args:
            tokens: The tokens to convert to IDs.

        Returns:
            The ID of the tokenized text
        """

        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if self.max_seq_length is not None:
            # Truncate to max sequence length.
            ids = ids[:self.max_seq_length]
            # Pad to max sequence length.
            pad_ids_to_add = self.max_seq_length - len(tokens)
            ids += [self.tokenizer.pad_token_id] * pad_ids_to_add
        return ids
