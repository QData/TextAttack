"""

Dataset Class
======================

TextAttack allows users to provide their own dataset or load from HuggingFace.


"""

from collections import OrderedDict
import random

import torch


class Dataset(torch.utils.data.Dataset):
    """Basic class for dataset. It operates as a map-style dataset, fetching
    data via :meth:`__getitem__` and :meth:`__len__` methods.

    .. note::
        This class subclasses :obj:`torch.utils.data.Dataset` and therefore can be treated as a regular PyTorch Dataset.

    Args:
        dataset (:obj:`list[tuple]`):
            A list of :obj:`(input, output)` pairs.
            If :obj:`input` consists of multiple fields (e.g. "premise" and "hypothesis" for SNLI),
            :obj:`input` must be of the form :obj:`(input_1, input_2, ...)` and :obj:`input_columns` parameter must be set.
            :obj:`output` can either be an integer representing labels for classification or a string for seq2seq tasks.
        input_columns (:obj:`list[str]`, `optional`, defaults to :obj:`["text"]`):
            List of column names of inputs in order.
        label_map (:obj:`dict[int, int]`, `optional`, defaults to :obj:`None`):
            Mapping if output labels of the dataset should be re-mapped. Useful if model was trained with a different label arrangement.
            For example, if dataset's arrangement is 0 for `Negative` and 1 for `Positive`, but model's label
            arrangement is 1 for `Negative` and 0 for `Positive`, passing :obj:`{0: 1, 1: 0}` will remap the dataset's label to match with model's arrangements.
            Could also be used to remap literal labels to numerical labels (e.g. :obj:`{"positive": 1, "negative": 0}`).
        label_names (:obj:`list[str]`, `optional`, defaults to :obj:`None`):
            List of label names in corresponding order (e.g. :obj:`["World", "Sports", "Business", "Sci/Tech"]` for AG-News dataset).
            If not set, labels will printed as is (e.g. "0", "1", ...). This should be set to :obj:`None` for non-classification datasets.
        output_scale_factor (:obj:`float`, `optional`, defaults to :obj:`None`):
            Factor to divide ground-truth outputs by. Generally, TextAttack goal functions require model outputs between 0 and 1.
            Some datasets are regression tasks, in which case this is necessary.
        shuffle (:obj:`bool`, `optional`, defaults to :obj:`False`): Whether to shuffle the underlying dataset.

            .. note::
                Generally not recommended to shuffle the underlying dataset. Shuffling can be performed using DataLoader or by shuffling the order of indices we attack.

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

    def filter_by_labels_(self, labels_to_keep):
        """Filter items by their labels for classification datasets. Performs
        in-place filtering.

        Args:
            labels_to_keep (:obj:`Union[Set, Tuple, List, Iterable]`):
                Set, tuple, list, or iterable of integers representing labels.
        """
        if not isinstance(labels_to_keep, set):
            labels_to_keep = set(labels_to_keep)
        self._dataset = list(filter(lambda x: x[1] in labels_to_keep, self._dataset))

    def __getitem__(self, i):
        """Return i-th sample."""
        if isinstance(i, int):
            return self._format_as_dict(self._dataset[i])
        else:
            # `idx` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return [self._format_as_dict(ex) for ex in self._dataset[i]]

    def __len__(self):
        """Returns the size of dataset."""
        return len(self._dataset)
