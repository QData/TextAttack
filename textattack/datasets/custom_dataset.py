import collections
import random

import datasets
import pandas as pd

import textattack
from textattack.datasets import TextAttackDataset


def _cb(s):
    """Colors some text blue for printing to the terminal."""
    if not isinstance(s, str):
        s = "custom " + str(type(s))
    return textattack.shared.utils.color_text(str(s), color="blue", method="ansi")


class CustomDataset(TextAttackDataset):
    """Loads a Custom Dataset from a file/list of files and prepares it as a
    TextAttack dataset.

    - name(Union[str, dict, pd.DataFrame]): the user specified dataset file names, dicts or pandas dataframe
    - infile_format(str): Specifies type of file for loading HuggingFaceDataset : csv, json, pandas, text
      from local_files will be loaded as ``datasets.load_dataset(filetype, data_files=name)``.
    - label_map: Mapping if output labels should be re-mapped. Useful
      if model was trained with a different label arrangement than
      provided in the ``datasets`` version of the dataset.
    - output_scale_factor (float): Factor to divide ground-truth outputs by.
        Generally, TextAttack goal functions require model outputs
        between 0 and 1. Some datasets test the model's correlation
        with ground-truth output, instead of its accuracy, so these
        outputs may be scaled arbitrarily.
    - dataset_columns (list): dataset_columns[0]: input columns specified as a tuple or list, dataset_columns[1]: output_columns
    - shuffle (bool): Whether to shuffle the dataset on load.
    """

    def __init__(
        self,
        name,
        infile_format=None,
        split="train",
        label_map=None,
        subset=None,
        output_scale_factor=None,
        dataset_columns=None,
        shuffle=False,
    ):

        self._name = name

        if infile_format in ["csv", "json", "text", "pandas"]:
            self._dataset = datasets.load_dataset(infile_format, data_files=self._name)[
                split
            ]

        else:
            if isinstance(self._name, dict):
                self._dataset = datasets.Dataset.from_dict(self._name)[split]
            elif isinstance(self._name, pd.DataFrame):
                self._dataset = datasets.Dataset.from_pandas(self._name)[split]
            else:
                raise ValueError(
                    "Only accepts csv, json, text, pandas file infile_format or dicts and pandas DataFrame"
                )

        subset_print_str = f", subset {_cb(subset)}" if subset else ""

        textattack.shared.logger.info(
            f"Loading {_cb('datasets')} dataset {_cb(name)}{subset_print_str}, split {_cb(split)}."
        )
        # Input/output column order, like (('premise', 'hypothesis'), 'label')

        if dataset_columns is None:
            # automatically infer from dataset
            dataset_columns = []
            dataset_columns.append(self._dataset.column_names)

        if not (
            isinstance(dataset_columns[0], list)
            or isinstance(dataset_columns[0], tuple)
        ):
            dataset_columns[0] = [dataset_columns[0]]

        if not set(dataset_columns[0]) <= set(self._dataset.column_names):
            raise ValueError(
                f"Could not find input column {dataset_columns[0]}. Found keys: {self._dataset.column_names}"
            )
        self.input_columns = dataset_columns[0]

        if len(dataset_columns) == 1:
            # if user hasnt specified an output column or dataset_columns is None, all dataset_columns are
            # treated as input_columns
            dataset_columns.append(None)
            self.output_column = dataset_columns[1]
        if (
            dataset_columns[1] is not None
            and dataset_columns[1] not in self._dataset.column_names
        ):
            # if user has specified an output column, user can specify output column as None
            raise ValueError(
                f"Could not find output column {dataset_columns[1]}. Found keys: {self._dataset.column_names}"
            )

        self._i = 0
        self.examples = list(self._dataset)

        self.label_map = label_map

        self.output_scale_factor = output_scale_factor

        try:

            self.label_names = self._dataset.features["label"].names

            # If labels are remapped, the label names have to be remapped as
            # well.
            if label_map:

                self.label_names = [
                    self.label_names[self.label_map[i]]
                    for i in range(len(self.label_map))
                ]

        except KeyError:
            # This happens when the dataset doesn't have 'features' or a 'label' column.

            self.label_names = None
        except AttributeError:
            # This happens when self._dataset.features["label"] exists
            # but is a single value.
            self.label_names = ("label",)

        if shuffle:
            random.shuffle(self.examples)

    def _format_raw_example(self, raw_example):
        input_dict = collections.OrderedDict(
            [(c, raw_example[c]) for c in self.input_columns]
        )
        if self.output_column is not None:
            output = raw_example[self.output_column]
            if self.label_map:
                output = self.label_map[output]
            if self.output_scale_factor:
                output = output / self.output_scale_factor
            return (input_dict, None)

        else:
            return (input_dict,)

    def __next__(self):
        if self._i >= len(self.examples):
            raise StopIteration
        raw_example = self.examples[self._i]
        self._i += 1
        return self._format_raw_example(raw_example)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._format_raw_example(self.examples[i])
        else:
            # `i` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return [self._format_raw_example(ex) for ex in self.examples[i]]
