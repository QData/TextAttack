class Perturbation:
    """ An abstract class for perturbing a string of text to produce
        a potential adversarial example. """
    def perturb(self, tokenized_text):
        """ Returns a list of all possible perturbations for `text`."""
        raise NotImplementedException()