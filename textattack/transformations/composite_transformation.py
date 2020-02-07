import numpy as np
from .transformation import Transformation

class CompositeTransformation(Transformation):
    def __init__(self, transformations):
        if not isinstance(transformations, list):
            raise TypeError('transformations must be a list')
        if not len(transformations):
            raise ValueError('transformations cannot be empty')
        self.transformations = transformations
    
    def __call__(self, *args, **kwargs):
        new_tokenized_texts = set()
        for transformation in self.transformations:
            new_tokenized_texts.update(
                transformation(*args, **kwargs)
            )
        return list(new_tokenized_texts)
        