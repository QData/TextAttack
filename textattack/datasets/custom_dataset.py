import collections
import random

import datasets

import textattack
from textattack.datasets import HuggingFaceDataset


def _cb(s):
    """Colors some text blue for printing to the terminal."""
    return textattack.shared.utils.color_text(str(s), color="blue", method="ansi")


class CustomDataset(HuggingFaceDataset):
    """Loads a Custom Dataset like HuggingFace custom ``datasets`` and prepares it as a
    TextAttack dataset.

    - name(str): the dataset file names
    - file_type(str): Specifies type of file for loading HuggingFaceDataset
      from local_files will be loaded as ``datasets.load_dataset(filetype, data_files=name)``.
    - label_map: Mapping if output labels should be re-mapped. Useful
      if model was trained with a different label arrangement than
      provided in the ``datasets`` version of the dataset.
    - output_scale_factor (float): Factor to divide ground-truth outputs by.
        Generally, TextAttack goal functions require model outputs
        between 0 and 1. Some datasets test the model's correlation
        with ground-truth output, instead of its accuracy, so these
        outputs may be scaled arbitrarily.
    - dataset_columns (list): dataset_columns[0]: input columns, dataset_columns[1]: output_columns
    - shuffle (bool): Whether to shuffle the dataset on load.
    """

    def __init__(
        self,
        name,
        filetype="csv",
        split="train",
        label_map=None,
        output_scale_factor=None,
        dataset_columns=[("text",), None],
        shuffle=False,
    ):

        self._name = name

        self._dataset = datasets.load_dataset(filetype, data_files=name)[split]

        subset = None
        subset_print_str = f", subset {_cb(subset)}" if subset else ""

        textattack.shared.logger.info(
            f"Loading {_cb('datasets')} dataset {_cb(name)}{subset_print_str}, split {_cb(split)}."
        )
        # Input/output column order, like (('premise', 'hypothesis'), 'label')
        if not set(dataset_columns[0]) <= set(self._dataset.column_names):
            raise ValueError(
                f"Could not find input column {dataset_columns[0]} in CSV. Found keys: {self._dataset.column_names}"
            )
        self.input_columns = dataset_columns[0]
        self.output_column = dataset_columns[1]
        if (
            self.output_column is not None
            and self.output_column not in self._dataset.column_names
        ):
            raise ValueError(
                f"Could not find input column {dataset_columns[1]} in CSV. Found keys: {self._dataset.column_names}"
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
                print(self.label_names)
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
        else:
            output = None

        return (input_dict, output)
