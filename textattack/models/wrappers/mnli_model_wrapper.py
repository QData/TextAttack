from .model_wrapper import ModelWrapper
from typing import Tuple
import torch
from torch.nn.functional import softmax

class MnliModelWrapper(ModelWrapper):
    def __init__(self, model):
        self.model = model

    def __call__(self, input_texts: Tuple[str]):
        ret = []
        for t in input_texts:
            premise = t[0]
            hypothesis = t[1]
            tokens = self.model.encode(premise, hypothesis)
            predict = self.model.predict('mnli', tokens)
            probs = softmax(predict, dim=1).cpu().detach()[0]
            ret.append(probs.unsqueeze(0))
        return ret