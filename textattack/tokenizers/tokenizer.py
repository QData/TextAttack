class Tokenizer:
    """ A generic class that convert text to tokens and tokens to IDs. Supports
        any type of tokenization, be it word, wordpiece, or character-based.
    """
    def convert_text_to_tokens(self, text):
        raise NotImplementedError()
        
    def convert_tokens_to_ids(self, ids):
        raise NotImplementedError()
        
    def encode(self, text):
        """ Converts text directly to IDs. """
        tokens = self.convert_text_to_tokens(text)
        return self.convert_tokens_to_ids(tokens)

    def replace_tokens(self, old_tokens, new_tokens, start_end_indices):
        """
        Take old_tokens and replaced items from selected start and end indicies with new_tokens
        """
        raise NotImplementedError

    def replace_ids(self, old_ids, new_ids, start_end_indices):
        """
        Take old_tokens and replaced items from selected start and end indicies with new_tokens
        """
        raise NotImplementedError