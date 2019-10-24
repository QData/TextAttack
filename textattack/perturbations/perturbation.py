import numpy as np

from textattack.constraints import Constraint


class Perturbation:
    """
    Abstract class for perturbing a string of text with the potential of
    creating an adversial example. 

    Args:
        constraints (:obj:`list`, optional): A list of constraints. Defaults to None. 
    
    
    """
    def __init__(self, constraints=[]):
        self.constraints = []
        if constraints:
            self.add_constraints(constraints)
    
    """
    An abstract class for perturbing a string of text to produce
    a potential adversarial example. 
    """
    def perturb(self, tokenized_text):
        """
        Returns a list of all possible perturbations for `text`
        that match provided constraints.

        Args:
            tokenized_test:

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()
    
    def _filter_perturbations(self, original_text, perturbations):
        """
        Filters a list of perturbations based on initalized constraints. 

        Args:
            original_text (str): The unaltered text
            perturbations (list): The list of potential perturbations
            
        """
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
        """
        Add one constraint to the attack. 

        Args:
            constraint: A constraint to add to the perturbation (see TODO)
        
        """
        self.constraints.append(constraint)
    
    def add_constraints(self, constraints):
        """ 
        Add multiple constraints to the attack.

        Args:
            constraints: A list of constraints to add to the perturbation (see TODO)

        Raises:
            TypeError: If :obj:`constraints` are not iterable.

        """
        # Make sure constraints are iterable.
        try:
            iter(constraints)
        except TypeError as te:
            raise TypeError(f'Constraint list type {type(constraints)} is not iterable.')
        # Store each constraint after validating its type.
        for constraint in constraints:
            if not isinstance(constraint, Constraint):
                raise ValueError('Cannot add constraint of type', type(constraint))
            self.add_constraint(constraint)
    
    def _filter_perturbations(self, original_text, perturbations):
        """ Filters a list of perturbations by self.constraints. """
        perturbations = np.array(perturbations)
        for C in self.constraints:
            perturbations = C.call_many(original_text, perturbations)
        return perturbations
