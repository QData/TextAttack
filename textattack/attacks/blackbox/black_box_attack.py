import torch

from textattack.attacks import Attack

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
        self.model_description = model.__class__.__name__
        self.model = model.__call__
        self.tokenizer = model.tokenizer
        super().__init__(constraints=constraints)
        
    def _call_model(self, *args, **kwargs):
        with torch.no_grad():
            return super()._call_model(*args, **kwargs)
