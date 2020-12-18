"""

datasets:
======================

TextAttack allows users to provide their own dataset or load from HuggingFace.


"""

from .dataset import Dataset, IterableDataset
from .huggingface_dataset import HuggingFaceDataset

from . import translation
