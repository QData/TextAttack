from .model_wrapper import ModelWrapper
from typing import Tuple, List
import torch
from torch.nn.functional import softmax

class FairseqMnliWrapper(ModelWrapper):
    """
    A wrapper for the model
        torch.hub.load('pytorch/fairseq', 'roberta.large.mnli').eval()
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, input_texts: List[Tuple[str, str]]) -> List[torch.Tensor]:
        """
        Args:
            input_texts: List[Tuple[str, str]]
                List of (premise, hypothesis)
        
        Return:
            ret: List[torch.Tensor]
                Each tensor is a list of probabilities, 
                one for each of (contradiction, neutral, entailment)
        """
        ret = []
        for t in input_texts:
            premise = t[0]
            hypothesis = t[1]
            tokens = self.model.encode(premise, hypothesis)
            predict = self.model.predict('mnli', tokens)
            probs = softmax(predict, dim=1).cpu().detach()[0]
            ret.append(probs.unsqueeze(0))
        return ret