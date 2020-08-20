import collections
import random

import nlp

import textattack
from textattack.datasets import TextAttackDataset

# from textattack.shared import AttackedText


def _cb(s):
    """Colors some text blue for printing to the terminal."""
    return textattack.shared.utils.color_text(str(s), color="blue", method="ansi")


def get_nlp_dataset_columns(dataset):
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
            f"Unsupported dataset schema {schema}. Try loading dataset manually (from a file) instead."
        )

    return input_columns, output_column


class HuggingFaceNlpDataset(TextAttackDataset):
    """Loads a dataset from HuggingFace ``nlp`` and prepares it as a TextAttack
    dataset.

    - name: the dataset name
    - subset: the subset of the main dataset. Dataset will be loaded as ``nlp.load_dataset(name, subset)``.
    - label_map: Mapping if output labels should be re-mapped. Useful
      if model was trained with a different label arrangement than
      provided in the ``nlp`` version of the dataset.
    - output_scale_factor (float): Factor to divide ground-truth outputs by.
        Generally, TextAttack goal functions require model outputs
        between 0 and 1. Some datasets test the model's correlation
        with ground-truth output, instead of its accuracy, so these
        outputs may be scaled arbitrarily.
    - shuffle (bool): Whether to shuffle the dataset on load.
    """

    def __init__(
        self,
        name,
        subset=None,
        split="train",
        label_map=None,
        output_scale_factor=None,
        dataset_columns=None,
        shuffle=False,
    ):
        self._name = name
        self._dataset = nlp.load_dataset(name, subset)[split]
        subset_print_str = f", subset {_cb(subset)}" if subset else ""
        textattack.shared.logger.info(
            f"Loading {_cb('nlp')} dataset {_cb(name)}{subset_print_str}, split {_cb(split)}."
        )
        # Input/output column order, like (('premise', 'hypothesis'), 'label')
        (
            self.input_columns,
            self.output_column,
        ) = dataset_columns or get_nlp_dataset_columns(self._dataset)
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

        output = raw_example[self.output_column]
        if self.label_map:
            output = self.label_map[output]
        if self.output_scale_factor:
            output = output / self.output_scale_factor

        return (input_dict, output)

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
