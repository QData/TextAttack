from constraints import TextConstraint

class Perturbation:
    """ Generates perturbations for a given text input. """
    def __init__(self, constraints=[]):
        print('Perturbation init')
        self.constraints = []
        if constraints:
            self.add_constraints(constraints)
    
    """ An abstract class for perturbing a string of text to produce
        a potential adversarial example. """
    def perturb(self, tokenized_text):
        """ Returns a list of all possible perturbations for `text`
            that match provided constraints."""
        raise NotImplementedError()
    
    def _filter_perturbations(self, original_text, perturbations):
        """ Filters a list of perturbations based on internal constraints. """
        good_perturbations = []
        for perturbed_text in perturbations:
            meets_constraints = True
            for c in self.constraints:
                meets_constraints = meets_constraints and c(original_text, 
                    perturbed_text)
                if not meets_constraints: break
            if meets_constraints: good_perturbations.append(p)
        return good_perturbations
    
    def add_constraint(self, constraint):
        """ Add constraint to attack. """
        self.constraints.append(constraint)
    
    def add_constraints(self, constraints):
        """ Add multiple constraints.
        """
        for constraint in constraints:
            if not isinstance(constraint, TextConstraint):
                raise ValueError('Cannot add constraint of type', type(constraint))
            self.add_constraint(constraint)