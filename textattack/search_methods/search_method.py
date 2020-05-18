from textattack.shared.utils import default_class_repr

class SearchMethod:
    """
    This is an abstract class that contains main helper functionality for 
    search methods.

    """
    def __call__(self, intial_result):
        """
        Perturbs `tokenized_text` from intial_result until goal is reached
        """
        raise NotImplementedError()

    def check_transformation_compatibility(self, transformation):
        return True

    def extra_repr_keys(self):
        return []
 
    __repr__ = __str__ = default_class_repr
