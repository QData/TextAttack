class Constraint:
    """ A constraint that evaluates if x_adv meets a certain constraint. """
    def __call__(self, x, x_adv):
        """ Returns True if C(x,x_adv) is true. """
        raise NotImplementedException()