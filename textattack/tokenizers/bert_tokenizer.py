from transformers.tokenization_bert import BertTokenizer, BertTokenizerFast
from textattack.tokenizers import Tokenizer

class BERTTokenizer(Tokenizer):
    """ A generic class that convert text to tokens and tokens to IDs. Supports
        any type of tokenization, be it word, wordpiece, or character-based.
    """
    def __init__(self, model_path='bert-base-uncased', max_seq_length=256, fast=False):
        self.max_seq_length = max_seq_length
        if fast:
            # Faster tokenizer that is implemented in Rust
            self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)

        self.max_seq_length = max_seq_length
        self.fast = fast

    def convert_text_to_tokens(self, input_text):
        """ 
        Takes a string input, tokenizes, formats, and returns a tensor with text 
        IDs. 
        
        Args:
            input_text (str): The text to tokenize

        Returns:
            The ID of the tokenized text
        """


        tokens = self.tokenizer.tokenize(input_text)
        if self.fast:
            # When using BertTokenizerFast, CLS and SEP tokens are inserted already
            tokens = tokens[:self.max_seq_length]
        else:
            tokens = tokens[:self.max_seq_length-2]
            tokens.insert(0, self.tokenizer.cls_token)
            tokens.append(self.tokenizer.sep_token)
        pad_tokens_to_add = self.max_seq_length - len(tokens)
        tokens += [self.tokenizer.pad_token] * pad_tokens_to_add
        return tokens
    
    def convert_tokens_to_ids(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids
