""" Abstract classes that represent constraints on text adversarial examples. 

    @TODO: Smart ordering of constraints:
        On each pass of call_many, time constraints and order them from fastest 
        to slowest. That way, the fastest constraint does the most work, then 
        subsequent constraints only have to examine perturbations that met 
        previous constraints.

"""


class Constraint:
    """ 
    An abstract class that represents constraints on adversial text examples. 
    A constraint evaluates if (x,x_adv) meets a certain constraint. 

    """
    
    def filter(self, x, x_adv_list):
        """
        Filters x_adv_list to x_adv where C(x,x_adv) is true.

        Args:
            x:
            x_adv_list:

        Returns:
            TODO
            
        @TODO can we just call this `filter`? My syntax highlighter highlights
            that so I'm inclined not to use that protected name...
        """
        raise NotImplementedError()
    
    def __call__(self, x, x_adv):
        """ Returns True if C(x,x_adv) is true. """
        raise NotImplementedError()