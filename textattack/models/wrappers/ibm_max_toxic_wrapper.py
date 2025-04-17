import numpy as np
from textattack.models.wrappers import ModelWrapper
from typing import List

class IBMMAXToxicWrapper(ModelWrapper):
    """
    A wrapper for the IBM Max Toxic model
    https://github.com/IBM/MAX-Toxic-Comment-Classifier/blob/master/core/model.py
    """
    def __init__(self, ibm_model_wrapper):
        """
        Args:
            ibm_model_wrapper: An instance of the IBM MAX Toxic `ModelWrapper()` class.
        """
        self.model = ibm_model_wrapper

    def __call__(self, input_text_list: List[str]) -> np.ndarray:
        """
        Args:
            input_texts: List[str]
        
        Return:
            ret: np.ndarray
                One entry per element in input_text_list. Each is a list of logits, one for each label.
        """
        self.model._pre_process(input_text_list)
        logits = self.model._predict(input_text_list)
        return np.array(logits)
