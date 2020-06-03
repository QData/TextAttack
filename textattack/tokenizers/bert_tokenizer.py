from textattack.tokenizers import AutoTokenizer

class BERTTokenizer(AutoTokenizer):
    """ 
    A generic class that convert text to tokens and tokens to IDs. Intended
    for fine-tuned BERT models.
    """
    def __init__(self, name='bert-base-uncased', max_seq_length=256):
        super().__init__(name, max_seq_length=max_seq_length)

    def convert_text_to_tokens(self, input_text):
        """ 
        Takes a string input, tokenizes, formats, and returns a list of tokens.
        
        BERT requires a special token to demarcate the beginning and end of each
        input sequence.
        
        Args:
            input_text (str): The text to tokenize

        Returns:
            The  list of tokens.
        """
        tokens = self.tokenizer.tokenize(input_text)
        tokens = tokens[:self.max_seq_length-2]
        tokens.insert(0, self.tokenizer.cls_token)
        tokens.append(self.tokenizer.sep_token)
        pad_tokens_to_add = self.max_seq_length - len(tokens)
        tokens += [self.tokenizer.pad_token] * pad_tokens_to_add
        return tokens
