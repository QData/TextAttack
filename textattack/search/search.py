class Search:
    """ This is an abstract class that defines a textual search method.
        `self.__call__` takes the original label and text of a text input
        and searches for a potential adversarial example. 
        
        Generally, a search method queries `get_transformations` to explore 
        candidate perturbed phrases.
    """
    def __call__(self, original_label, tokenized_text):
        raise NotImplementedError()