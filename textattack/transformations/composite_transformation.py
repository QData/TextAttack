import numpy as np

from textattack.transformations.transformation import Transformation


class CompositeTransformation(Transformation):
    """
    A transformation which applies each of a list of transformations, returning a set of 
    all optoins.

    Args:
        transformations: The list of ``Transformation``\s to apply.
    """

    def __init__(self, transformations):
        if not (
            isinstance(transformations, list) or isinstance(transformations, tuple)
        ):
            raise TypeError("transformations must be list or tuple")
        elif not len(transformations):
            raise ValueError("transformations cannot be empty")
        self.transformations = transformations

    def __call__(self, *args, **kwargs):
        new_attacked_texts = set()
        for transformation in self.transformations:
            new_attacked_texts.update(transformation(*args, **kwargs))
        return list(new_attacked_texts)
