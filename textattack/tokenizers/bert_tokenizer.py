from transformers.tokenization_bert import BertTokenizer
from textattack.tokenizers import Tokenizer

class BERTTokenizer(Tokenizer):
    """ A generic class that convert text to tokens and tokens to IDs. Supports
        any type of tokenization, be it word, wordpiece, or character-based.
    """
    def __init__(self, model_path='bert-base-uncased', max_seq_length=256):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_length = max_seq_length
        
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
        tokens = tokens[:self.max_seq_length-2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        pad_tokens_to_add = self.max_seq_length - len(tokens)
        tokens += [self.tokenizer.pad_token] * pad_tokens_to_add
        return tokens
    
    def convert_tokens_to_ids(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids