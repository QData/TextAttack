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


class Dataset:
    """Basic class for dataset. It operates as a map-style dataset, fetching
    data via `__getitem__` and `__len__` methods. For datasets that fetch data
    via `__iter__` protocol should be created using `IterableDataset` class.

    Args:
        data (list_like): A list-like iterable of ``(input, output)`` pairs. Here, `output` can either be an integer representing labels for classification
            or a string for seq2seq tasks. If input consists of multiple sequences (e.g. SNLI), iterable
            should be of the form ``([input_1, input_2, ...], output)`` and ``input_columns`` parameter must be set.
        lang (str, optional): Two letter ISO 639-1 code representing the language of the input data (e.g. "en", "fr", "ko", "zh"). Default is "en".
        input_columns (list[str], optional): List of column names of inputs in order. Default is ``["text"]`` for single text input.
        label_names (list[str], optional): List of label names in corresponding order (e.g. ``["World", "Sports", "Business", "Sci/Tech"] for AG-News dataset).
            If not set, labels will printed as is (e.g. "0", "1", ...). This should be set to ``None`` for non-classification datasets.
        shuffle (bool): Whether to shuffle the dataset on load.

    Examples::

        >>> import textattack

        >>> # Example of sentiment-classification dataset
        >>> data = [("I enjoyed the movie a lot!", 1), ("Absolutely horrible film.", 0), ("Our family had a fun time!", 1)]
        >>> dataset = textattack.datasets.Dataset(data, lang="en")
        >>> dataset[1:2]


        >>> # Example for pair of sequence inputs (e.g. SNLI)
        >>> data = [("A man inspects the uniform of a figure in some East Asian country.", "The man is sleeping"), 1)]
        >>> dataset = textattack.datasets.Dataset(data, lang="en", input_columns=("premise", "hypothesis"))

        >>> # Example for seq2seq
        >>> data = [("J'aime le film.", "I love the movie.")]
        >>> dataset = textattack.datasets.Dataset(data, lang="fr")
    """

    def __init__(
        self, data, lang="en", input_columns=["text"], label_names=None, shuffle=False
    ):
        self._data = data
        self.lang = lang
        self.input_columns = input_columns
        self.label_names = label_names
        self.shuffled = shuffle

        if shuffle:
            random.shuffle(self.data)

    def _format_as_dict(self, example):
        output = example[1]
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
        random.shuffle(self.data)
        self.shuffled = True

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._format_as_dict(self.data[i])
        else:
            # `i` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return [self._format_as_dict(ex) for ex in self.data[i]]

    def __len__(self):
        return len(self.data)
