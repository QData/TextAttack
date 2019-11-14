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
        self.text_to_ids_converter = model.convert_text_to_ids
        super().__init__(constraints=constraints)
        
    def _call_model(self, tokenized_text_list):
        with torch.no_grad():
            super()._call_model(tokenized_text_list)