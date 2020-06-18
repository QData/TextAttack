from textattack.shared.utils import default_class_repr


class Constraint:
    """ 
    An abstract class that represents constraints on adversial text examples. 
    Constraints evaluate whether transformations from a ``AttackedText`` to another 
    ``AttackedText`` meet certain conditions.
    """

    def call_many(self, transformed_texts, current_text, original_text=None):
        """
        Filters ``transformed_texts`` based on which transformations fulfill the constraint.
        First checks compatibility with latest ``Transformation``, then calls 
        ``_check_constraint_many``\.

        Args:
            transformed_texts: The candidate transformed ``AttackedText``\s.
            current_text: The current ``AttackedText``.
            original_text: The original ``AttackedText`` from which the attack began.
        """
        incompatible_transformed_texts = []
        compatible_transformed_texts = []
        for transformed_text in transformed_texts:
            try:
                if self.check_compatibility(
                    transformed_text.attack_attrs["last_transformation"]
                ):
                    compatible_transformed_texts.append(transformed_text)
                else:
                    incompatible_transformed_texts.append(transformed_text)
            except KeyError:
                raise KeyError(
                    "transformed_text must have `last_transformation` attack_attr to apply constraint"
                )
        filtered_texts = self._check_constraint_many(
            compatible_transformed_texts, current_text, original_text=original_text
        )
        return list(filtered_texts) + incompatible_transformed_texts

    def _check_constraint_many(
        self, transformed_texts, current_text, original_text=None
    ):
        """
        Filters ``transformed_texts`` based on which transformations fulfill the constraint.
        Calls ``check_constraint``\.

        Args:
            transformed_texts: The candidate transformed ``AttackedText``\s.
            current_text: The current ``AttackedText``.
            original_text: The original ``AttackedText`` from which the attack began.
        """
        return [
            transformed_text
            for transformed_text in transformed_texts
            if self._check_constraint(
                transformed_text, current_text, original_text=original_text
            )
        ]

    def __call__(self, transformed_text, current_text, original_text=None):
        """ 
        Returns True if the constraint is fulfilled, False otherwise. First checks
        compatibility with latest ``Transformation``, then calls ``_check_constraint``\.
        
        Args:
            transformed_text: The candidate transformed ``AttackedText``.
            current_text: The current ``AttackedText``.
            original_text: The original ``AttackedText`` from which the attack began.
        """
        if not isinstance(transformed_text, AttackedText):
            raise TypeError("transformed_text must be of type AttackedText")
        if not isinstance(current_text, AttackedText):
            raise TypeError("current_text must be of type AttackedText")

        try:
            if not self.check_compatibility(
                transformed_text.attack_attrs["last_transformation"]
            ):
                return True
        except KeyError:
            raise KeyError(
                "x_adv must have `last_transformation` attack_attr to apply constraint."
            )
        return self._check_constraint(
            transformed_text, current_text, original_text=original_text
        )

    def _check_constraint(self, transformed_text, current_text, original_text=None):
        """ 
        Returns True if the constraint is fulfilled, False otherwise. Must be overridden by
        the specific constraint.
        
        Args:
            transformed_text: The candidate transformed ``AttackedText``.
            current_text: The current ``AttackedText``.
            original_text: The original ``AttackedText`` from which the attack began.
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
