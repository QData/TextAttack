from textattack.shared import TokenizedText
from textattack.tokenizers import BERTTokenizer

class BERTEntailmentTokenizer(BERTTokenizer):
    """ 
    Tokenizes an input for entailment. 
    """
    def __init__(self, name='bert-base-uncased'): 
        super().__init__(name=name)
        
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
                
    def convert_text_to_tokens(self, entailment_input):
        """ 
        Takes a string input, tokenizes, formats, and returns a list of text tokens.
        
        Args:
            entailment_input (string): A dictionary containing the premise and
                hypothesis, with a '|' in between them.

        Returns:
            A list of text tokens.
        """
        # Get tokenized premise and hypothesis.
        premise, hypothesis = entailment_input.split(TokenizedText.SPLIT_TOKEN)
        tokens_a = self.tokenizer.tokenize(premise)
        tokens_b = self.tokenizer.tokenize(hypothesis)
        # Ensure they will fit in self.max_seq_length.
        self._truncate_seq_pair(tokens_a, tokens_b)
        # Concatenate and return.
        return ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
    
    
    def convert_tokens_to_ids(self, tokens):
        """ Takes in a tokenized (premise, hypothesis) pair and returns the
            three ID vectors. 
        """
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
        
        
