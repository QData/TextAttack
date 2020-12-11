"""
Transformation Abstract Class
============================================

"""

from abc import ABC, abstractmethod

from textattack.shared.utils import default_class_repr


class Transformation(ABC):
    """An abstract class for transforming a sequence of text to produce a
    potential adversarial example."""

    def __call__(
        self,
        current_text,
        pre_transformation_constraints=[],
        indices_to_modify=None,
        shifted_idxs=False,
    ):
        """Returns a list of all possible transformations for ``current_text``.
        Applies the ``pre_transformation_constraints`` then calls
        ``_get_transformations``.

        Args:
            current_text: The ``AttackedText`` to transform.
            pre_transformation_constraints: The ``PreTransformationConstraint`` to apply before
                beginning the transformation.
            indices_to_modify: Which word indices should be modified as dictated by the
                ``SearchMethod``.
            shifted_idxs (bool): Whether indices could have been shifted from
                their original position in the text.
        """
        if indices_to_modify is None:
            indices_to_modify = set(range(len(current_text.words)))
            # If we are modifying all indices, we don't care if some of the indices might have been shifted.
            shifted_idxs = False
        else:
            indices_to_modify = set(indices_to_modify)

        if shifted_idxs:
            indices_to_modify = set(
                current_text.convert_from_original_idxs(indices_to_modify)
            )

        for constraint in pre_transformation_constraints:
            indices_to_modify = indices_to_modify & constraint(current_text, self)
        transformed_texts = self._get_transformations(current_text, indices_to_modify)
        for text in transformed_texts:
            text.attack_attrs["last_transformation"] = self
        return transformed_texts

    @abstractmethod
    def _get_transformations(self, current_text, indices_to_modify):
        """Returns a list of all possible transformations for ``current_text``,
        only modifying ``indices_to_modify``. Must be overridden by specific
        transformations.

        Args:
            current_text: The ``AttackedText`` to transform.
            indicies_to_modify: Which word indices can be modified.
        """
        raise NotImplementedError()

    @property
    def deterministic(self):
        return True

    def extra_repr_keys(self):
        return []

    __repr__ = __str__ = default_class_repr
