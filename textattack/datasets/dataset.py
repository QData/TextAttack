"""
dataset: TextAttack dataset
=============================
Dataset classes define the dataset object used to for carrying out attacks, augmentation, and training.
``Dataset`` class is the most basic class that could be used to wrap a list of input and output pairs.
To load datasets from text, CSV, or JSON files, we recommend using Huggingface's `datasets` library to first
load it as a `datasets.Dataset` object and then pass it to TextAttack's `HuggingFaceDataset` class.
Lastly, if you need to load data via the `__iter__` protocol, you can extend the `IterableDataset` class.
"""
from abc import ABC, abstractmethod
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
        lang (str): Two letter ISO 639-1 code representing the language of the input data (e.g. "en", "fr", "ko", "zh"). Default is "en".
        input_columns (list[str]): List of column names of inputs in order. Default is ``["text"]`` for single text input.

    Examples::

        >>> import textattack

        >>> # Example of sentiment-classification dataset
        >>> data = [("I enjoyed the movie a lot!", 1), ("Absolutely horrible film.", 0), ("Our family had a fun time!", 1)]
        >>> dataset = textattack.dataset.Dataset(data, lang="en")
        >>> dataset[1:2]


        >>> # Example for pair of sequence inputs (e.g. SNLI)
        >>> data = [("A man inspects the uniform of a figure in some East Asian country.", "The man is sleeping"), 1)]
        >>> dataset = textattack.dataset.Dataset(data, lang="en", input_columns=("premise", "hypothesis"))

        >>> # Example for seq2seq
        >>> data = [("J'aime le film.", "I love the movie.")]
        >>> dataset = textattack.dataset.Dataset(data, lang="fr")
    """

    def __init__(self, data, lang="en", input_columns=["text"], shuffle=False):
        self.data = data
        self.lang = lang
        self.input_columns = input_columns
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

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._format_example(self.data[i])
        else:
            # `i` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return [self._format_example(ex) for ex in self.data[i]]

    def __len__(self):
        return len(self.data)


class IterableDataset(ABC):
    """Basic class for datasets that fetch data via ``__iter__`` protocol. Idea
    is similar to PyTorch's ``IterableDataset``. This is useful if you cannot
    load the entire dataset to memory, such as reading from a large file.
    Unlike ``Dataset``, ``IterableDataset` is an abstract base class, meaning
    that you need to extend it with your own custom child class and define the
    ``format_example`` method. This is to suppport flexible preprocessing of
    each example returned by the underlying iterator.

    However, for most cases that involve loading from a txt, CSV, or JSON files, we recommend loading it as Huggingface's ``datasets.Dataset`` object and
    pass it to TextAttack's ``HuggingFaceDataset`` class. This class is designed for cases where you really need to load data via ``__iter__`` protocol.

    For more information about loading files as ``datasets.Dataset`` object, visit https://huggingface.co/docs/datasets/loading_datasets.html.

    Args:
        data_iterator: Iterator that returns next element when ``next(data_iterator)`` is called.
        lang (str): Two letter ISO 639-1 code representing the language of the data (e.g. "en", "fr", "ko", "zh"). Default is "en".
        input_columns (list[str]): List of column names of inputs in order. Default is ``["text"]`` for single text input.

    Examples::
        Suppose `data.csv` looks like the following:
            Text, Sentiment
            "I enjoyed the movie a lot!",Positive
            "Absolutely horrible film.",Negative
            "Our family had a fun time!",Positive

        Here is an example of how to load the file using `IterableDataset`. Note that we want to turn labels into corresponding integers.
        >>> import csv
        >>> import textattack

        >>> class MyIterableDataset(textattack.datasets.IterableDataset):
        ...     def format_example(self, example):
        ...         label = 1 if example[1] == "Positive" else 0
        ...         return example[0], label
        >>> f = open("data.csv")
        >>> reader = csv.reader(f)
        >>> next(reader) # Skip the first header line
        >>> dataset = MyIterableDataset(reader, lang="en")
        >>> for row in dataset:
        ...     print(row)
        (OrderedDict([('text', 'I enjoyed the movie a lot!')]), 1)
        (OrderedDict([('text', 'Absolutely horrible film.')]), 0)
        (OrderedDict([('text', 'Our family had a fun time!')]), 1)
    """

    def __init__(self, data_iterator, lang="en", input_columns=["text"]):
        self.data_iterator = data_iterator
        self.lang = lang
        self.input_columns = input_columns

        self._i = 0

    def _format_as_dict(self, example):
        output = example[1]
        if isinstance(example[0], str):
            input_dict = OrderedDict([(self.input_columns[0], example[0])])
        else:
            input_dict = OrderedDict(
                [(c, example[0][i]) for i, c in enumerate(self.input_columns)]
            )
        return input_dict, output

    @abstractmethod
    def format_example(self, raw_example):
        """Receives raw example returned by underlying iterator and formats it
        as a ```(input, output)`` pair.

        Here, `output` can either be an integer representing labels for
        classification or a string for seq2seq tasks. If input consists
        of multiple sequences (e.g. SNLI), the returned pair should be
        of the form ``([input_1, input_2, ...], output)`` where inputs
        are ordered as defined by the input_columns` attribute of the
        dataset.
        """
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self):
        example = next(self.data_iterator)
        formatted_example = self.format_example(example)
        return self._format_as_dict(formatted_example)
