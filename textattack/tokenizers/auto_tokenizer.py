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
    def __init__(self, name='bert-base-uncased', max_seq_length=256, use_fast=True):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(name, use_fast=use_fast)
        self.max_seq_length = max_seq_length
        
    def _truncate_seq_pair(self, tokens_a, tokens_b):
        """ 
        Truncates a sequence pair in place to the maximum length.

        This is a simple heuristic which will always truncate the longer 
        sequence one token at a time. This makes more sense than truncating an
        equal percent of tokens from each, since if one sequence is very short
        then each token that's truncated likely contains more information than 
        a longer sequence.
        """
        max_length = self.max_seq_length - 3 # Subtract 3 for 'CLS' and 2 'SEP' tokens
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= self.max_seq_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

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
            return ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
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
        
        if tokens.count('[SEP]') == 2: # multi-input
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            
            # Segment IDs should have 0s for the premise, 1s for the hypothesis,
            # and then pad with 0s after.
            premise_length = tokens.index('[SEP]')
            hypothesis_length = len(tokens) - premise_length
            segment_ids = ([0] * premise_length) + ([1] * hypothesis_length)
            
            # Add padding up to self.max_seq_length.
            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
    
            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            
            return input_ids, input_mask, segment_ids
        else: # single input
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            if self.max_seq_length is not None:
                # Truncate to max sequence length.
                ids = ids[:self.max_seq_length]
                # Pad to max sequence length.
                pad_ids_to_add = self.max_seq_length - len(tokens)
                ids += [self.tokenizer.pad_token_id] * pad_ids_to_add
            return ids