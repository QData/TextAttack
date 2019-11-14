from transformers.modeling_bert import BertForSequenceClassification
from transformers.tokenization_bert import BertTokenizer

import textattack.utils as utils
import torch

class BERTForClassification:
    """ 
    BERT fine-tuned for sentiment classification. 

    Args:
        max_seq_length(:obj:`string`): Path to the pre-trained model.
        max_seq_length(:obj:`int`, optional):  Number of class labels for 
            prediction, if different than 2.
        max_seq_length(:obj:`int`, optional):  Maximum length of a sequence after tokenizing.
            Defaults to 32.
            
    """
    def __init__(self, model_path, num_labels=2, max_seq_length=64):
        utils.download_if_needed(model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model.to(utils.get_device())
        self.model.eval()
        self.max_seq_length = max_seq_length
    
    def convert_text_to_ids(self, input_text):
        """ 
        Takes a string input, tokenizes, formats,
        and returns a tensor with text IDs. 
        
        Args:
            input_text (str): The text to tokenize

        Returns:
            The ID of the tokenized text
        """
        tokens = self.tokenizer.tokenize(input_text)
        while len(tokens) > self.max_seq_length:
            tokens.pop()
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        pad_tokens_to_add = self.max_seq_length + 2 - len(tokens)
        tokens += [self.tokenizer.pad_token] * pad_tokens_to_add
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids
    
    def __call__(self, text_ids):
        if not isinstance(text_ids, torch.Tensor):
            raise ValueError(f'Object of type {type(text_ids)} must be of type torch.tensor')
        pred = self.model(text_ids)
        return pred[0]
