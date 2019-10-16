""" Abstract classes that represent constraints on text adversarial examples. """
class Constraint:
    """ A constraint that evaluates if (x,x_adv) meets a certain constraint. """
    
    def call_many(self, x, x_adv_list):
        """ Filters x_adv_list to x_adv where C(x,x_adv) is true.
            
            @TODO can we just call this `filter`? My syntax highlighter highlights
                that so I'm inclined not to use that protected name...
        """
        raise NotImplementedError()
    
    def __call__(self, x, x_adv):
        """ Returns True if C(x,x_adv) is true. """
        raise NotImplementedError()