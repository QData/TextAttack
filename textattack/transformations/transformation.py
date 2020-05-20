from textattack.shared.utils import default_class_repr

class Transformation:
    """
    An abstract class for transforming a sequence of text to produce
    a potential adversarial example. 
        
    """

    def __call__(self, tokenized_text, pre_transformation_constraints=[], indices_to_modify=None):
        """ 
        Returns a list of all possible transformations for ``tokenized_text``. Applies the
        ``pre_transformation_constraints`` then calles ``_get_transformations``.

        Args:
            tokenized_text: The ``TokenizedText`` to transform.
            pre_transformation_constraints: The ``PreTransformationConstraint``\s to apply before
                beginning the transformation.
            indicies_to_modify: Which word indices should be modified as dictated by the
                ``SearchMethod``.
        """
        if indices_to_modify is None:
            indices_to_modify = set(range(len(tokenized_text.words)))
        else:
            indices_to_modify = set(indices_to_modify)
        for constraint in pre_transformation_constraints:
            indices_to_modify = indices_to_modify & constraint(tokenized_text, self)
        transformed_texts = self._get_transformations(tokenized_text, indices_to_modify)
        for text in transformed_texts:
            text.attack_attrs['last_transformation'] = self
        return transformed_texts
   
    def _get_transformations(self, tokenized_text, indices_to_modify):
        """ 
        Returns a list of all possible transformations for ``tokenized_text``, only modifying
        ``indices_to_modify``. Must be overridden by specific transformations.

        Args:
            tokenized_text: The ``TokenizedText`` to transform.
                beginning the transformation.
            indicies_to_modify: Which word indices can be modified.
        """
        raise NotImplementedError()

    def extra_repr_keys(self): 
        return []

    __repr__ = __str__ = default_class_repr
