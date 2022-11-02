"""

Ted Multi TranslationDataset Class
------------------------------------
"""


import collections

import datasets
import numpy as np

from textattack.datasets import HuggingFaceDataset


class TedMultiTranslationDataset(HuggingFaceDataset):
    """Loads examples from the Ted Talk translation dataset using the
    `datasets` package.

    dataset source: http://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/
    """

    def __init__(self, source_lang="en", target_lang="de", split="test", shuffle=False):
        self._dataset = datasets.load_dataset("ted_multi")[split]
        self.examples = self._dataset["translations"]
        language_options = set(self.examples[0]["language"])
        if source_lang not in language_options:
            raise ValueError(
                f"Source language {source_lang} invalid. Choices: {sorted(language_options)}"
            )
        if target_lang not in language_options:
            raise ValueError(
                f"Target language {target_lang} invalid. Choices: {sorted(language_options)}"
            )
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.shuffled = shuffle
        self.label_map = None
        self.output_scale_factor = None
        self.label_names = None
        # self.input_columns = ("Source",)
        # self.output_column = "Translation"

        if shuffle:
            self._dataset.shuffle()

    def _format_as_dict(self, raw_example):
        example = raw_example["translations"]
        translations = np.array(example["translation"])
        languages = np.array(example["language"])
        source = translations[languages == self.source_lang][0]
        target = translations[languages == self.target_lang][0]
        source_dict = collections.OrderedDict([("Source", source)])
        return (source_dict, target)
