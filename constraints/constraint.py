""" Abstract classes that represent constraints on text adversarial examples. """
class Constraint:
    """ A constraint that evaluates if (x,x_adv) meets a certain constraint. """
    def __call__(self, x, x_adv):
        """ Returns True if C(x,x_adv) is true. """
        raise NotImplementedError()

class WordConstraint(Constraint):
    """ A constraint that evaluates if an (original, perturbed) word pair
        meets a certain constraint. """
    def __call__(self, x, x_adv):
        """ Returns True if C(x,x_adv) is true. """
        raise NotImplementedError()


class TextConstraint(Constraint):
    """ A constraint that evaluates if an (original, perturbed) text input pair
        meets a certain constraint. """
    def __call__(self, x, x_adv):
        """ Returns True if C(x,x_adv) is true. """
        raise NotImplementedError()

