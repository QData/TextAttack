from .model_wrapper import ModelWrapper
from typing import List

class PipelineModelWrapper(ModelWrapper):
    """
    A general model wrapper for Hugging Face Pipeline models. 
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, input_texts: List[str]):
        """
        Args:
            input_texts: List[str]
        
        Return:
            ret
                Model output
        """
        ret = []
        for i in input_texts:
            pred = self.model(i)
            ret.append(pred)
        return ret