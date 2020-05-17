""" Abstract classes represent constraints on text adversarial examples. 
"""

from textattack.shared.utils import default_class_repr

class Constraint:
    """ 
    An abstract class that represents constraints on adversial text examples. 
    A constraint evaluates if (x,x_adv) meets a certain constraint. 

    """
     
    def call_many(self, x, x_adv_list, original_text=None):
        """
        Filters x_adv_list to x_adv where C(x,x_adv) is true.

        Args:
            x:
            x_adv_list:
            original_text(:obj:`type`, optional): Defaults to None. 

        """
        incompatible_x_advs = []
        compatible_x_advs = []
        for x_adv in x_adv_list:
            try:
                if self.check_compatibility(x_adv.attack-attrs['last_transformation']):
                    compatible_x_advs.append(x_adv)
                else:
                    incompatible_x_advs.append(x_adv)
            except KeyError:
                raise KeyError('x_adv must have `last_transformation` attack_attr to apply GoogLM constraint')
        filtered_x_advs = self._check_constraint_many(x, compatible_x_advs, original_text=original_text)
        return filtered_x_advs + incompatible_x_advs

    def _check_constraint_many(self, x, x_adv_list, original_text=None):
        return [x_adv for x_adv in x_adv_list 
                if self._check_constraint(x, x_adv, original_text=original_text)]

    def __call__(self, x, x_adv, original_text=None):
        """ Returns True if C(x,x_adv) is true. """
        if not isinstance(x, TokenizedText):
            raise TypeError('x must be of type TokenizedText')
        if not isinstance(x_adv, TokenizedText):
            raise TypeError('x_adv must be of type TokenizedText')

        try:
            if not self.check_compatibility(x_adv.attack_attrs['last_transformation']):
                return True
        except KeyError:
            raise KeyError('x_adv must have `last_transformation` attack_attr to apply constraint.')
        return self._check_constraint(x_adv, original_text=original_text)

    def _check_constraint(self, x, x_adv, original_text=None):
        """ Returns True if C(x,x_adv) is true. """
        raise NotImplementedError()

    def check_compatibility(self, transformation):
        """ 
        Checks if this constraint is compatible with the given transformation.
        Args:
            transformation: The transformation to check compatibility with.
        """
        return True

    def extra_repr_keys(self):
        """Set the extra representation of the constraint using these keys.
        
        To print customized extra information, you should reimplement
        this method in your own constraint. Both single-line and multi-line
        strings are acceptable.
        """ 
        return []
    
    __str__ = __repr__ = default_class_repr
