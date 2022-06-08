"""

HuggingFaceDataset Class
=========================

TextAttack allows users to provide their own dataset or load from HuggingFace.


"""

import collections

import datasets

import textattack

from .dataset import Dataset


def _cb(s):
    """Colors some text blue for printing to the terminal."""
    return textattack.shared.utils.color_text(str(s), color="blue", method="ansi")


def get_datasets_dataset_columns(dataset):
    """Common schemas for datasets found in dataset hub."""
    schema = set(dataset.column_names)
    if {"premise", "hypothesis", "label"} <= schema:
        input_columns = ("premise", "hypothesis")
        output_column = "label"
    elif {"question", "sentence", "label"} <= schema:
        input_columns = ("question", "sentence")
        output_column = "label"
    elif {"sentence1", "sentence2", "label"} <= schema:
        input_columns = ("sentence1", "sentence2")
        output_column = "label"
    elif {"question1", "question2", "label"} <= schema:
        input_columns = ("question1", "question2")
        output_column = "label"
    elif {"question", "sentence", "label"} <= schema:
        input_columns = ("question", "sentence")
        output_column = "label"
    elif {"context", "question", "title", "answers"} <= schema:
        # Common schema for SQUAD dataset
        input_columns = ("title", "context", "question")
        output_column = "answers"
    elif {"text", "label"} <= schema:
        input_columns = ("text",)
        output_column = "label"
    elif {"sentence", "label"} <= schema:
        input_columns = ("sentence",)
        output_column = "label"
    elif {"document", "summary"} <= schema:
        input_columns = ("document",)
        output_column = "summary"
    elif {"content", "summary"} <= schema:
        input_columns = ("content",)
        output_column = "summary"
    elif {"label", "review"} <= schema:
        input_columns = ("review",)
        output_column = "label"
    else:
        raise ValueError(
            f"Unsupported dataset schema {schema}. Try passing your own `dataset_columns` argument."
        )

    return input_columns, output_column


class HuggingFaceDataset(Dataset):
    """Loads a dataset from 🤗 Datasets and prepares it as a TextAttack dataset.

    Args:
        name_or_dataset (:obj:`Union[str, datasets.Dataset]`):
            The dataset name as :obj:`str` or actual :obj:`datasets.Dataset` object.
            If it's your custom :obj:`datasets.Dataset` object, please pass the input and output columns via :obj:`dataset_columns` argument.
        subset (:obj:`str`, `optional`, defaults to :obj:`None`):
            The subset of the main dataset. Dataset will be loaded as :obj:`datasets.load_dataset(name, subset)`.
        split (:obj:`str`, `optional`, defaults to :obj:`"train"`):
            The split of the dataset.
        dataset_columns (:obj:`tuple(list[str], str))`, `optional`, defaults to :obj:`None`):
            Pair of :obj:`list[str]` representing list of input column names (e.g. :obj:`["premise", "hypothesis"]`)
            and :obj:`str` representing the output column name (e.g. :obj:`label`). If not set, we will try to automatically determine column names from known designs.
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
    """

    def __init__(
        self,
        name_or_dataset,
        subset=None,
        split="train",
        dataset_columns=None,
        label_map=None,
        label_names=None,
        output_scale_factor=None,
        shuffle=False,
    ):
        if isinstance(name_or_dataset, datasets.Dataset):
            self._dataset = name_or_dataset
        else:
            self._name = name_or_dataset
            self._subset = subset
            self._dataset = datasets.load_dataset(self._name, subset)[split]
            subset_print_str = f", subset {_cb(subset)}" if subset else ""
            textattack.shared.logger.info(
                f"Loading {_cb('datasets')} dataset {_cb(self._name)}{subset_print_str}, split {_cb(split)}."
            )
        # Input/output column order, like (('premise', 'hypothesis'), 'label')
        (
            self.input_columns,
            self.output_column,
        ) = dataset_columns or get_datasets_dataset_columns(self._dataset)

        if not isinstance(self.input_columns, (list, tuple)):
            raise ValueError(
                "First element of `dataset_columns` must be a list or a tuple."
            )

        self.label_map = label_map
        self.output_scale_factor = output_scale_factor
        if label_names:
            self.label_names = label_names
        else:
            try:
                self.label_names = self._dataset.features[self.output_column].names
            except (KeyError, AttributeError):
                # This happens when the dataset doesn't have 'features' or a 'label' column.
                self.label_names = None

        # If labels are remapped, the label names have to be remapped as well.
        if self.label_names and label_map:
            self.label_names = [
                self.label_names[self.label_map[i]] for i in self.label_map
            ]

        self.shuffled = shuffle
        if shuffle:
            self._dataset.shuffle()

    def _format_as_dict(self, example):
        input_dict = collections.OrderedDict(
            [(c, example[c]) for c in self.input_columns]
        )

        output = example[self.output_column]
        if self.label_map:
            output = self.label_map[output]
        if self.output_scale_factor:
            output = output / self.output_scale_factor

        return (input_dict, output)

    def filter_by_labels_(self, labels_to_keep):
        """Filter items by their labels for classification datasets. Performs
        in-place filtering.

        Args:
            labels_to_keep (:obj:`Union[Set, Tuple, List, Iterable]`):
                Set, tuple, list, or iterable of integers representing labels.
        """
        if not isinstance(labels_to_keep, set):
            labels_to_keep = set(labels_to_keep)
        self._dataset = self._dataset.filter(
            lambda x: x[self.output_column] in labels_to_keep
        )

    def __getitem__(self, i):
        """Return i-th sample."""
        if isinstance(i, int):
            return self._format_as_dict(self._dataset[i])
        else:
            # `idx` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return [
                self._format_as_dict(self._dataset[j]) for j in range(i.start, i.stop)
            ]

    def shuffle(self):
        self._dataset.shuffle()
        self.shuffled = True
