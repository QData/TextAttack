from abc import ABC, abstractmethod

from textattack.shared.utils import default_class_repr


class PreTransformationConstraint(ABC):
    """An abstract class that represents constraints which are applied before
    the transformation.

    These restrict which words are allowed to be modified during the
    transformation. For example, we might not allow stopwords to be
    modified.
    """

    def __call__(self, current_text, transformation):
        """Returns the word indices in ``current_text`` which are able to be
        modified. First checks compatibility with ``transformation`` then calls
        ``_get_modifiable_indices``

        Args:
            current_text: The ``AttackedText`` input to consider.
            transformation: The ``Transformation`` which will be applied.
        """
        if not self.check_compatibility(transformation):
            return set(range(len(current_text.words)))
        return self._get_modifiable_indices(current_text)

    @abstractmethod
    def _get_modifiable_indices(current_text):
        """Returns the word indices in ``current_text`` which are able to be
        modified. Must be overridden by specific pre-transformation
        constraints.

        Args:
            current_text: The ``AttackedText`` input to consider.
        """
        raise NotImplementedError()

    def check_compatibility(self, transformation):
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

    def extra_repr_keys(self):
        """Set the extra representation of the constraint using these keys.

        To print customized extra information, you should reimplement
        this method in your own constraint. Both single-line and multi-
        line strings are acceptable.
        """
        return []

    __str__ = __repr__ = default_class_repr
