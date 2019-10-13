from transformers.modeling_bert import BertForSequenceClassification
from transformers.tokenization_bert import BertTokenizer

import utils
import torch

class BertForSentimentClassification:
    """ BERT fine-tuned on the Yelp Sentiment dataset for sentiment classification. """
    
    MODEL_PATH = '/p/qdata/jm8wx/research/RobustNLP/AttackGeneration/models/bert/models/bert-vanilla'
    
    def __init__(self, max_seq_length=32):
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
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        while len(ids) < self.max_seq_length + 2:
            ids = ids + [0] # @TODO Is it correct to just pad with zeros?
        return ids
    
    def __call__(self, text_ids):
        if not isinstance(text_ids, torch.Tensor):
            raise ValueError(f'Object of type {type(text_ids)} must be of type torch.tensor')
        pred = self.model(text_ids)
        return pred[0]

# Rewrite 'SimpleBertClassifier' from RobustNLP using new transformers package.