import torch

from textattack.attacks import Attack

import torch

class BlackBoxAttack(Attack):
    """ An abstract class that defines a black-box attack. 
    
        
        A black-box attack can access the prediction scores of a model, but
        not any other information about its parameters or internal state. We
        reduce the model here to just the __call__ function so that black-box
        attacks that extend this class can obtain prediction scores, but not
        any other information from the model.
        
        Arg:
            model (nn.Module): The model to attack.
            constraints (list(Constraint)): The list of constraints
                for the model's transformations.
    
    """
    def __init__(self, model, constraints=[]):
        self.model = model.__call__
        self.text_to_tokens_converter = model.convert_text_to_tokens
        self.tokens_to_ids_converter = model.convert_tokens_to_ids
        super().__init__(constraints=constraints)
        
    def _call_model(self, *args, **kwargs):
        with torch.no_grad():
            return super()._call_model(*args, **kwargs)
