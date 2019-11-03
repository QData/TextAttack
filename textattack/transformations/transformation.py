class Transformation:
    """
    An abstract class for transofrming a string of text to produce
    a potential adversarial example. 
        
    """

    def __call__(self, tokenized_text):
        """ Returns a list of all possible transformations for `text`."""
        raise NotImplementedError()
