from textattack.shared.utils import default_class_repr

class Transformation:
    """
    An abstract class for transofrming a string of text to produce
    a potential adversarial example. 
        
    """

    def __call__(self, tokenized_text, modification_constraints=[]):
        """ Returns a list of all possible transformations for `tokenized_text`."""
        modifiable_indices = set(range(len(tokenized_text.words)))
        for constraint in modification_constraints:
            if constraint.check_compatibility(self):
                modifiable_indices = modifiable_indices & constraint(tokenized_text, self)
        transformed_texts = _get_transformations(tokenized_text, modifiable_indices)
        for text in transformed_texts:
            text.attack_attrs['last_transformation'] = self
        return transformed_texts
   
    def _get_transformations(self, tokenized_text, modifiable_indices)
        raise NotImplementedError()

    def extra_repr_keys(self): 
        return []
        
    __repr__ = __str__ = default_class_repr
