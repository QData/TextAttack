import spacy
class SpacyTokenizer:
    """ A basic implementation of the spaCy English tokenizer. 
    
        Params:
            word2id (dict<string, int>): A dictionary that matches words to IDs
            oovid (int): An out-of-variable ID
    """
    def __init__(self, word2id, oovid, padid):
        self.tokenizer = spacy.load('en').tokenizer
        self.word2id = word2id
        self.oovid = oovid
        self.padid = padid
    
    def convert_text_to_tokens(self, text):
        spacy_tokens = english_tokenizer(text)
        return [t.text for t in spacy_tokens]
        
    def convert_tokens_to_ids(self, tokens):
        ids = []
        for word in input_tokens:
            if word in self.word2id:
                ids.append(self.word2id[word])
            else:
                ids.append(self.oovid)
        return ids
        
    def encode(text):
        """ Converts text directly to IDs. """
        tokens = self.convert_text_to_tokens(text)
        return self.convert_tokens_to_ids(tokens)