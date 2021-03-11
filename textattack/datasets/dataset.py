"""
dataset: TextAttack dataset
=============================
Dataset classes define the dataset object used to for carrying out attacks, augmentation, and training.
``Dataset`` class is the most basic class that could be used to wrap a list of input and output pairs.
To load datasets from text, CSV, or JSON files, we recommend using Huggingface's `datasets` library to first
load it as a `datasets.Dataset` object and then pass it to TextAttack's `HuggingFaceDataset` class.
"""
from collections import OrderedDict
import random

import torch


class Dataset(torch.utils.data.Dataset):
    """Basic class for dataset. It operates as a map-style dataset, fetching
    data via `__getitem__` and `__len__` methods.

    Args:
        dataset (list_like): A list-like iterable of ``(input, output)`` pairs. Here, `output` can either be an integer representing labels for classification
            or a string for seq2seq tasks. If input consists of multiple sequences (e.g. SNLI), iterable should be of the form ``([input_1, input_2, ...], output)`` and ``input_columns`` parameter must be set.
        input_columns (list[str], optional): List of column names of inputs in order. Default is ``["text"]`` for single text input.
        label_map (dict, optional): Mapping if output labels of the dataset should be re-mapped. Useful if model was trained with a different label arrangement than
            provided in the ``datasets`` version of the dataset. For example, if dataset's arrangement is 0 for negative and 1 for positive, but model's label
            arrangement is 1 for negative and 0 for positive, pass ``{0: 1, 1: 0}``. Could also be used to remap literal labels to numerical labels,
            (e.g. ``{"positive": 1, "negative": 0}``)
        label_names (list[str], optional): List of label names in corresponding order (e.g. ``["World", "Sports", "Business", "Sci/Tech"] for AG-News dataset).
            If not set, labels will printed as is (e.g. "0", "1", ...). This should be set to ``None`` for non-classification datasets.
        output_scale_factor (float): Factor to divide ground-truth outputs by. Generally, TextAttack goal functions require model outputs between 0 and 1. Some datasets are regression tasks, in which case this is necessary.
        shuffle (bool): Whether to shuffle the dataset on load.

    Examples::

        >>> import textattack

        >>> # Example of sentiment-classification dataset
        >>> data = [("I enjoyed the movie a lot!", 1), ("Absolutely horrible film.", 0), ("Our family had a fun time!", 1)]
        >>> dataset = textattack.datasets.Dataset(data)
        >>> dataset[1:2]


        >>> # Example for pair of sequence inputs (e.g. SNLI)
        >>> data = [("A man inspects the uniform of a figure in some East Asian country.", "The man is sleeping"), 1)]
        >>> dataset = textattack.datasets.Dataset(data, input_columns=("premise", "hypothesis"))

        >>> # Example for seq2seq
        >>> data = [("J'aime le film.", "I love the movie.")]
        >>> dataset = textattack.datasets.Dataset(data)
    """

    def __init__(
        self,
        dataset,
        input_columns=["text"],
        label_map=None,
        label_names=None,
        output_scale_factor=None,
        shuffle=False,
    ):
        self._dataset = dataset
        self.input_columns = input_columns
        self.label_map = label_map
        self.label_names = label_names
        if label_map:
            # If labels are remapped, the label names have to be remapped as well.
            self.label_names = [
                self.label_names[self.label_map[i]] for i in self.label_map
            ]
        self.shuffled = shuffle
        self.output_scale_factor = output_scale_factor

        if shuffle:
            random.shuffle(self._dataset)

    def _format_as_dict(self, example):
        output = example[1]
        if self.label_map:
            output = self.label_map[output]
        if self.output_scale_factor:
            output = output / self.output_scale_factor

        if isinstance(example[0], str):
            if len(self.input_columns) != 1:
                raise ValueError(
                    "Mismatch between the number of columns in `input_columns` and number of columns of actual input."
                )
            input_dict = OrderedDict([(self.input_columns[0], example[0])])
        else:
            if len(self.input_columns) != len(example[0]):
                raise ValueError(
                    "Mismatch between the number of columns in `input_columns` and number of columns of actual input."
                )
            input_dict = OrderedDict(
                [(c, example[0][i]) for i, c in enumerate(self.input_columns)]
            )
        return input_dict, output

    def shuffle(self):
        random.shuffle(self._dataset)
        self.shuffled = True

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._format_as_dict(self._dataset[i])
        else:
            # `i` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return [self._format_as_dict(ex) for ex in self._dataset[i]]

    def __len__(self):
        return len(self._dataset)
