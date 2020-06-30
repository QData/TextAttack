from abc import ABC, abstractmethod

import torch

class BaseModel(ABC):
    """ 
    An abstract base class models in TextAttack

    Args:
        model_path(:obj:`string`): Path to the pre-trained model.
        num_labels(:obj:`int`, optional):  Number of class labels for 
            prediction, if different than 2.
            
    """

    @abstractmethod
    def __call__(self, input_ids=None, **kwargs):
        raise NotImplementedError()
