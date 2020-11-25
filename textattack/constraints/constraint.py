"""

TextAttack Constraint Class
=====================================
"""

from abc import ABC, abstractmethod
from typing import List, Set

import textattack
from textattack.shared import AttackedText
from textattack.shared.utils import default_class_repr
from textattack.transformations import Transformation


class Constraint(ABC):
    """An abstract class that represents constraints on adversial text
    examples. Constraints evaluate whether transformations from a
    ``AttackedText`` to another ``AttackedText`` meet certain conditions.

    Args:
        compare_against_original (bool): If `True`, the reference text should be the original text under attack.
            If `False`, the reference text is the most recent text from which the transformed text was generated.
            All constraints must have this attribute.
    """

    def __init__(self, compare_against_original: bool):
        self.compare_against_original = compare_against_original

    def call_many(
        self, transformed_texts: List[AttackedText], reference_text: AttackedText
    ) -> List[AttackedText]:
        """Filters ``transformed_texts`` based on which transformations fulfill
        the constraint. First checks compatibility with latest
        ``Transformation``, then calls ``_check_constraint_many``

        Args:
            transformed_texts (list[AttackedText]): The candidate transformed ``AttackedText``'s.
            reference_text (AttackedText): The ``AttackedText`` to compare against.
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
            compatible_transformed_texts, reference_text
        )
        return list(filtered_texts) + incompatible_transformed_texts

    def _check_constraint_many(
        self, transformed_texts: List[AttackedText], reference_text: AttackedText
    ) -> List[AttackedText]:
        """Filters ``transformed_texts`` based on which transformations fulfill
        the constraint. Calls ``check_constraint``

        Args:
            transformed_texts (list[AttackedText]): The candidate transformed ``AttackedText``
            reference_texts (AttackedText): The ``AttackedText`` to compare against.
        """
        return [
            transformed_text
            for transformed_text in transformed_texts
            if self._check_constraint(transformed_text, reference_text)
        ]

    def __call__(
        self, transformed_text: AttackedText, reference_text: AttackedText
    ) -> bool:
        """Returns True if the constraint is fulfilled, False otherwise. First
        checks compatibility with latest ``Transformation``, then calls
        ``_check_constraint``

        Args:
            transformed_text (AttackedText): The candidate transformed ``AttackedText``.
            reference_text (AttackedText): The ``AttackedText`` to compare against.
        """
        if not isinstance(transformed_text, textattack.shared.AttackedText):
            raise TypeError("transformed_text must be of type AttackedText")
        if not isinstance(reference_text, textattack.shared.AttackedText):
            raise TypeError("reference_text must be of type AttackedText")

        try:
            if not self.check_compatibility(
                transformed_text.attack_attrs["last_transformation"]
            ):
                return True
        except KeyError:
            raise KeyError(
                "`transformed_text` must have `last_transformation` attack_attr to apply constraint."
            )
        return self._check_constraint(transformed_text, reference_text)

    @abstractmethod
    def _check_constraint(
        self, transformed_text: AttackedText, reference_text: AttackedText
    ) -> bool:
        """Returns True if the constraint is fulfilled, False otherwise. Must
        be overridden by the specific constraint.

        Args:
            transformed_text: The candidate transformed ``AttackedText``.
            reference_text (AttackedText): The ``AttackedText`` to compare against.
        """
        raise NotImplementedError()

    def check_compatibility(self, transformation: Transformation) -> bool:
        """Checks if this constraint is compatible with the given
        transformation. For example, the ``WordEmbeddingDistance`` constraint
        compares the embedding of the word inserted with that of the word
        deleted. Therefore it can only be applied in the case of word swaps,
        and not for transformations which involve only one of insertion or
        deletion.

        Args:
            transformation: The ``Transformation`` to check compatibility with.
        """
        return True

    def extra_repr_keys(self) -> List[str]:
        """Set the extra representation of the constraint using these keys.

        To print customized extra information, you should reimplement
        this method in your own constraint. Both single-line and multi-
        line strings are acceptable.
        """
        return ["compare_against_original"]

    __str__ = __repr__ = default_class_repr
