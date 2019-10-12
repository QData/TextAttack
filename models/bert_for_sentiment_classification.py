from transformers.modeling_bert import BertForSequenceClassification
from transformers.tokenization_bert import BertTokenizer

import utils

class BertForSentimentClassification:
    """ BERT fine-tuned on the Yelp Sentiment dataset for sentiment classification. """
    
    MODEL_PATH = '/p/qdata/jm8wx/research/RobustNLP/AttackGeneration/models/bert/models/bert-vanilla'
    
    def __init__(self, max_seq_length=128):
        utils.download_if_needed(BertForSentimentClassification.MODEL_PATH)
        
        self.model = BertForSequenceClassification.from_pretrained(
            BertForSentimentClassification.MODEL_PATH, 
            num_labels=2)
        self.tokenizer = BertTokenizer.from_pretrained(
            BertForSentimentClassification.MODEL_PATH)
        self.model.to(utils.get_device())
        self.model.eval()
        self.max_seq_length = max_seq_length
    
    def convert_text_to_ids(self, input_text):
        """ Takes a string input, tokenizes, formats,
            and returns a tensor with text IDs. """
        tokens = self.tokenizer.tokenize(input_text)
        while len(tokens) > self.max_seq_length:
            tokens.pop()
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        while len(tokens) < self.max_seq_length:
            tokens = tokens + [0]
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def __call__(self, text_ids):
        if not isinstance(text_ids, torch.tensor):
            raise ValueError(f'Object of type {type(text_ids)} must be of type torch.tensor')
        print('Bert input:', text_ids)
        pred = self.model(text_ids)
        # @TODO do we need to do crossentropy here?
        print('bert output:', pred)
        return pred

# Rewrite 'SimpleBertClassifier' from RobustNLP using new transformers package.