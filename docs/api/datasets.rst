Datasets API Reference
=============================
Dataset class define the dataset object used to for carrying out attacks, augmentation, and training.

Dataset
----------
:class:`~textattack.datasets.Dataset` class is the most basic class that could be used to wrap a list of input and output pairs.
To load datasets from text, CSV, or JSON files, we recommend using 🤗 Datasets library to first
load it as a :obj:`datasets.Dataset` object and then pass it to TextAttack's :class:`~textattack.datasets.HuggingFaceDataset` class.

.. autoclass:: textattack.datasets.Dataset
   :members:

HuggingFaceDataset
-------------------
.. autoclass:: textattack.datasets.HuggingFaceDataset
   :members: