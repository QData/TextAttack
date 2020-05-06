import spacy

from textattack.tokenizers import Tokenizer

class SpacyTokenizer(Tokenizer):
    """ A basic implementation of the spaCy English tokenizer. 
    
        Params:
            word2id (dict<string, int>): A dictionary that matches words to IDs
            oov_id (int): An out-of-variable ID
    """
    def __init__(self, word2id, oov_id, pad_id, max_seq_length=128):
        self.tokenizer = spacy.load('en').tokenizer
        self.word2id = word2id
        self.id2word = {v: k for k, v in word2id.items()}
        self.oov_id = oov_id
        self.pad_id = pad_id
        self.max_seq_length = max_seq_length
    
    def convert_text_to_tokens(self, text):
        spacy_tokens = [t.text for t in self.tokenizer(text)]
        return spacy_tokens[:self.max_seq_length]
        
    def convert_tokens_to_ids(self, tokens):
        ids = []
        for raw_token in tokens:
            token = raw_token.lower()
            if token in self.word2id:
                ids.append(self.word2id[token])
            else:
                ids.append(self.oov_id)
        pad_ids_to_add = [self.pad_id] * (self.max_seq_length - len(ids))
        ids += pad_ids_to_add
        return ids
    
    def convert_id_to_word(self, _id):
        """
        Takes an integer input and returns the corresponding word from the 
        vocabulary.
        
        Raises: KeyError on OOV.
        """
        return self.id2word[_id]