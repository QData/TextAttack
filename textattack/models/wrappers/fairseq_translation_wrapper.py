from textattack.models.wrappers import ModelWrapper
from typing import List

class FairseqTranslationWrapper(ModelWrapper):
    """
    A wrapper for the model
        torch.hub.load('pytorch/fairseq',
                        'transformer.wmt14.en-fr',
                        tokenizer='moses',
                        bpe='subword_nmt',
                        verbose=False).eval()
    or any other model with a .translate() method.
    """

    def __init__(self, model):
        self.model = model  

    def __call__(self, text_input_list: List[str]) -> List[str]:
        """
        Args:
            input_texts: List[str]
        
        Return:
            ret: List[str]
                Result of translation. One per element in input_texts.
        """
        return [self.model.translate(text) for text in text_input_list]
