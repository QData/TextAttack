from textattack.shared.utils import default_class_repr

class Constraint:
    """ 
    An abstract class that represents constraints on adversial text examples. 
    A constraint evaluates if (x,x_adv) meets a certain constraint. 

    """
     
    def call_many(self, x, x_adv_list, original_text=None):
        """
        Filters ``x_adv_list`` to ``x_adv`` where ``x_adv`` fulfills the constraint.
        First checks compatibility with latest ``Transformation``, then calls 
        ``_check_constraint_many``\.

        Args:
            x: The current ``TokenizedText``.
            x_adv_list: The potential altered ``TokenizedText``\s.
            original_text: The original ``TokenizedText`` from which the attack began.
        """
        incompatible_x_advs = []
        compatible_x_advs = []
        for x_adv in x_adv_list:
            try:
                if self.check_compatibility(x_adv.attack_attrs['last_transformation']):
                    compatible_x_advs.append(x_adv)
                else:
                    incompatible_x_advs.append(x_adv)
            except KeyError:
                raise KeyError('x_adv must have `last_transformation` attack_attr to apply constraint')
        filtered_x_advs = self._check_constraint_many(x, compatible_x_advs, original_text=original_text)
        return list(filtered_x_advs) + incompatible_x_advs

    def _check_constraint_many(self, x, x_adv_list, original_text=None):
        """
        Filters ``x_adv_list`` to ``x_adv`` where ``x_adv`` fulfills the constraint.
        Calls ``check_constraint``\.

        Args:
            x: The current ``TokenizedText``.
            x_adv_list: The potential altered ``TokenizedText``\s.
            original_text: The original ``TokenizedText`` from which the attack began.
        """
        incompatible_x_advs = []
        compatible_x_advs = []
        for x_adv in x_adv_list:
            try:
                if self.check_compatibility(x_adv.attack_attrs['last_transformation']):
                    compatible_x_advs.append(x_adv)
                else:
                    incompatible_x_advs.append(x_adv)
            except KeyError:
                raise KeyError('x_adv must have `last_transformation` attack_attr to apply constraint')
        filtered_x_advs = self._check_constraint_many(x, compatible_x_advs, original_text=original_text)
        return list(filtered_x_advs) + incompatible_x_advs

    def _check_constraint_many(self, x, x_adv_list, original_text=None):
        return [x_adv for x_adv in x_adv_list 
                if self._check_constraint(x, x_adv, original_text=original_text)]

    def __call__(self, x, x_adv, original_text=None):
        """ 
        Returns True if the constraint is fulfilled, False otherwise. First checks
        compatibility with latest ``Transformation``, then calls ``_check_constraint``\.
        
        Args:
            x: The current ``TokenizedText``.
            x_adv: The potential altered ``TokenizedText``.
            original_text: The original ``TokenizedText`` from which the attack began.
        """
        if not isinstance(x, TokenizedText):
            raise TypeError('x must be of type TokenizedText')
        if not isinstance(x_adv, TokenizedText):
            raise TypeError('x_adv must be of type TokenizedText')

        try:
            if not self.check_compatibility(x_adv.attack_attrs['last_transformation']):
                return True
        except KeyError:
            raise KeyError('x_adv must have `last_transformation` attack_attr to apply constraint.')
        return self._check_constraint(x_adv, original_text=original_text)

    def _check_constraint(self, x, x_adv, original_text=None):
        """ 
        Returns True if the constraint is fulfilled, False otherwise. Must be implemented
        by the specific constraint.
        
        Args:
            x: The current ``TokenizedText``.
            x_adv: The potential altered ``TokenizedText``.
            original_text: The original ``TokenizedText`` from which the attack began.
        """
        raise NotImplementedError()

    def check_compatibility(self, transformation):
        """ 
        Checks if this constraint is compatible with the given transformation.
        For example, the ``WordEmbeddingDistance`` constraint compares the embedding of
        the word inserted with that of the word deleted. Therefore it can only be
        applied in the case of word swaps, and not for transformations which involve
        only one of insertion or deletion.

        Args:
            transformation: The ``Transformation`` to check compatibility with.
        """
        return True

    def extra_repr_keys(self):
        """Set the extra representation of the constraint using these keys.
        
        To print customized extra information, you should reimplement
        this method in your own constraint. Both single-line and multi-line
        strings are acceptable.
        """ 
        return []
    
    __str__ = __repr__ = default_class_repr
