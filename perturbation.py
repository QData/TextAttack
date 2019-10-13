class Perturbation:
    """ Generates perturbations for a given text input. """
    
    def __init__(self, constraints=[]):
        if constraints:
            self.add_constraints(constraints)
    
    """ An abstract class for perturbing a string of text to produce
        a potential adversarial example. """
    def perturb(self, tokenized_text):
        """ Returns a list of all possible perturbations for `text`
            that match provided constraints."""
        raise NotImplementedError()
    
    def add_constraint(self, constraint):
        """ Add constraint to attack. """
        raise NotImplementedError()
    
    def add_constraints(self, constraints):
        """ Add multiple constraints.
        """
        for constraint in constraints:
            self.add_constraint(constraint)