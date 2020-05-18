""" Abstract classes represent constraints on text adversarial examples. 
"""

from textattack.shared.utils import default_class_repr
from textattack.constraints import Constraint

class ModificationConstraint(Constraint):
    """ 
    An abstract class that represents constraints which apply only 
    to which words can be modified. 
    """
   
    def __call__(self, x, transformation):
        """ Returns the word indices in x which are able to be modified """
        if not self.check_compatibility(transformation):
            return set(range(len(x.words)))
        return self._get_modifiable_indices(x)

    def _get_modifiable_indices(x):
        raise NotImplementedError()
