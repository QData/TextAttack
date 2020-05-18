from textattack.shared.utils import default_class_repr

class Transformation:
    """
    An abstract class for transofrming a string of text to produce
    a potential adversarial example. 
        
    """

    def __call__(self, tokenized_text, modification_constraints=[], indices_to_modify=None):
        """ Returns a list of all possible transformations for `tokenized_text`."""
        if indices_to_modify is None:
            indices_to_modify = set(range(len(tokenized_text.words)))
        else:
            indices_to_modify = set(indices_to_modify)
        for constraint in modification_constraints:
            indices_to_modify = indices_to_modify & constraint(tokenized_text, self)
        transformed_texts = self._get_transformations(tokenized_text, indices_to_modify)
        for text in transformed_texts:
            text.attack_attrs['last_transformation'] = self
        return transformed_texts
   
    def _get_transformations(self, tokenized_text, indices_to_modify):
        raise NotImplementedError()

    def extra_repr_keys(self): 
        return []

    def consists_of(self, subclass):
        return isinstance(self, subclass)
        
    __repr__ = __str__ = default_class_repr
