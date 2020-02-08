""" Abstract classes represent constraints on text adversarial examples. 
"""


class Constraint:
    """ 
    An abstract class that represents constraints on adversial text examples. 
    A constraint evaluates if (x,x_adv) meets a certain constraint. 

    """
    
    def call_many(self, x, x_adv_list, original_text=None, **kwargs):
        """
        Filters x_adv_list to x_adv where C(x,x_adv) is true.

        Args:
            x:
            x_adv_list:
            original_text(:obj:`type`, optional): Defaults to None. 

        """
        return [x_adv for x_adv in x_adv_list 
                if self.__call__(x, x_adv, original_text=original_text)]
    
    def __call__(self, x, x_adv, original_text=None):
        """ Returns True if C(x,x_adv) is true. """
        raise NotImplementedError()
