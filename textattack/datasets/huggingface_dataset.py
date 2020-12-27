"""
HuggingFaceDataset:
====================
"""

import collections
import random

import datasets

import textattack
from .dataset import Dataset

# from textattack.shared import AttackedText


def _cb(s):
    """Colors some text blue for printing to the terminal."""
    return textattack.shared.utils.color_text(str(s), color="blue", method="ansi")


def get_datasets_dataset_columns(dataset):
    """Common schemas for datasets found in dataset hub"""
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
    """Loads a dataset from HuggingFace ``datasets`` and prepares it as a
    TextAttack dataset.

    - name_or_dataset (Union[datasets.Dataset, str]): the dataset name or actual ``datasets.Dataset`` object. If it's your custom ``datasets.Dataset`` object,
        please pass the input and output columns via ``dataset_columns`` argument.
    - subset (str, optional): the subset of the main dataset. Dataset will be loaded as ``datasets.load_dataset(name, subset)``. Default is ``None``.
    - split (str, optioanl): the split of the dataset. Default is "train".
    - lang (str, optional): Two letter ISO 639-1 code representing the language of the input data (e.g. "en", "fr", "ko", "zh"). Default is "en".
    - dataset_columns (tuple(list[str], str)), optional): Pair of ``list[str]`` representing list of input column names (e.g. ["premise", "hypothesis"]) and
        ``str`` representing the output column name (e.g. ``label``). If not set, we will try to automatically determine column names from known designs. 
    - label_map (dict, optional): Mapping if output labels should be re-mapped. Useful if model was trained with a different label arrangement than
        provided in the ``datasets`` version of the dataset. For example, to remap "Positive" label to 1 and "Negative" label to 0, pass `{"Positive": 1, "Negative": 0}`.
    - label_names (list[str], optional): List of label names in corresponding order (e.g. ``["World", "Sports", "Business", "Sci/Tech"] for AG-News dataset).
        If ``datasets.Dataset`` object already has label names, then this is not required. Also, this should be set to ``None`` for non-classification datasets.
    - shuffle (bool): Whether to shuffle the dataset on load.
    """

    def __init__(
        self,
        name_or_dataset,
        subset=None,
        split="train",
        lang="en",
        dataset_columns=None,
        label_map=None,
        label_names=None,
        shuffle=False,
    ):
        if isinstance(name_or_dataset, datasets.Dataset):
            self._dataset = name_or_dataset
        else:
            self._name = name_or_dataset
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
        self.label_map = label_map
        try:
            self.label_names = self._dataset.features["label"].names
        except KeyError:
            # This happens when the dataset doesn't have 'features' or a 'label' column.
            self.label_names = None
        if shuffle:
            self._dataset.shuffle()

    def _format_as_dict(self, example):
        input_dict = collections.OrderedDict(
            [(c, example[c]) for c in self.input_columns]
        )

        output = example[self.output_column]
        if self.label_map:
            output = self.label_map[output]

        return (input_dict, output)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._format_as_dict(self._dataset[i])
        else:
            # `i` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return [self._format_as_dict(ex) for ex in self._dataset[i]]

    def __len__(self):
        return len(self._dataset)
