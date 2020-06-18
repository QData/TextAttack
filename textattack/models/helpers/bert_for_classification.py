import torch
from transformers.modeling_bert import BertForSequenceClassification
<<<<<<< HEAD

from textattack.shared import utils
from textattack.tokenizers import BERTTokenizer

=======
>>>>>>> 6953f0ee7d024957774d19d101175f0fa0176ccc

from textattack.models.tokenizers import BERTTokenizer
from textattack.shared import utils


class BERTForClassification:
    """ 
    BERT fine-tuned for textual classification. 

    Args:
        model_path(:obj:`string`): Path to the pre-trained model.
        num_labels(:obj:`int`, optional):  Number of class labels for 
            prediction, if different than 2.
            
    """

    def __init__(self, model_path, num_labels=2):
        model_file_path = utils.download_if_needed(model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            model_file_path, num_labels=num_labels
        )
<<<<<<< HEAD
        print("self", self, "to", utils.device)
=======
>>>>>>> 6953f0ee7d024957774d19d101175f0fa0176ccc
        self.model.to(utils.device)
        self.model.eval()
        self.tokenizer = BERTTokenizer(model_file_path)

    def __call__(self, input_ids=None, **kwargs):
        # The tokenizer will return ``input_ids`` along with ``token_type_ids``
        # and an ``attention_mask``. Our pre-trained models only need the input
        # IDs.
        pred = self.model(input_ids=input_ids)[0]
        return torch.nn.functional.softmax(pred, dim=-1)
