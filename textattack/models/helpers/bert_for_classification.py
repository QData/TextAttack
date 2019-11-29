import textattack.utils as utils
import torch

from textattack.tokenizers import BERTTokenizer
from transformers.modeling_bert import BertForSequenceClassification


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
    def __init__(self, model_path, num_labels=2):
        utils.download_if_needed(model_path)
        print('TextAttack BERTForClassification Loading from path', model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels)
        self.model.to(utils.get_device())
        self.model.eval()
        self.tokenizer = BERTTokenizer(model_path)
        self.word_embeddings = self.model.bert.embeddings.word_embeddings
        self.lookup_table = self.word_embeddings.weight.data
    
    def zero_grad(self): 
        self.model.zero_grad()
    
    def __call__(self, text_ids):
        if not isinstance(text_ids, torch.Tensor):
            raise ValueError(f'Object of type {type(text_ids)} must be of type torch.tensor')
        pred = self.model(text_ids)[0]
        return torch.nn.functional.softmax(pred, dim=-1)
