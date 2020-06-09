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
        
        
    
    def convert_text_to_tokens(self, input_text):
        """ 
        Takes a string input, tokenizes, formats, and returns a list of tokens.
        
        Args:
            input_text (str): The text to tokenize.

        Returns:
            A list of tokens.
        """
        if TokenizedText.SPLIT_TOKEN in input_text:
            # Tokenize both parts of input text.
            part_a, part_b = input_text.split(TokenizedText.SPLIT_TOKEN)
            tokens_a = self.tokenizer.tokenize(part_a)
            tokens_b = self.tokenizer.tokenize(part_b)
            # Ensure they will fit in self.max_seq_length.
            self._truncate_seq_pair(tokens_a, tokens_b)
            # Concatenate and return.
            tokens = tokens_a + ['[SEP]'] + tokens_b
        else:
            tokens = self.tokenizer.tokenize(input_text)
        return self._add_special_tokens(tokens)
    
    def convert_tokens_to_ids(self, tokens):
        """ 
        Takes a list of tokens and returns a tensor with text IDs. 
        
        Args:
            tokens: The tokens to convert to IDs.

        Returns:
            The ID of the tokenized text
        """
        import pdb; pdb.set_trace()
        
        if tokens.count('[SEP]') == 2: # multi-input
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            
            # Segment IDs should have 0s for the premise, 1s for the hypothesis,
            # and then pad with 0s after.
            premise_length = tokens.index('[SEP]')
            hypothesis_length = len(tokens) - premise_length
            token_type_ids = ([0] * premise_length) + ([1] * hypothesis_length)
            
            return { 'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids }
        else: # single input
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            if self.max_seq_length is not None:
                # Truncate to max sequence length.
                ids = ids[:self.max_seq_length]
                # Pad to max sequence length.
            return ids