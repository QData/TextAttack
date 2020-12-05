"""     
        
 dataset: TextAttack dataset        
 =============================      
 """


 from abc import ABC
import pickle
import random

import pandas as pd

from textattack.shared import utils


class TextAttackDataset(ABC):
    """Any iterable of (label, text_input) pairs qualifies as a
    ``TextAttackDataset``."""

    def __iter__(self):
        return self

    def _process_example_from_file(self, raw_line):
        """Processes each example read from a file. Implemented on a dataset-
        by-dataset basis.

        Args:
            raw_line (str): Line of the example to process.

        Returns:
            A tuple of text objects
        """
        raise NotImplementedError()

    def __next__(self):
        if self._i >= len(self.examples):
            raise StopIteration
        example = self.examples[self._i]
        self._i += 1
        return example

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)

    def _load_pickle_file(self, file_name, offset=0):
        self._i = 0
        file_path = utils.download_if_needed(file_name)
        with open(file_path, "rb") as f:
            self.examples = pickle.load(f)
        self.examples = self.examples[offset:]

    def _load_classification_text_file(self, text_file_name, offset=0, shuffle=False):
        """Loads tuples from lines of a classification text file.

        Format must look like:

            1 this is a great little ...
            0 "i love hot n juicy .  ...
            0 "\""this world needs a ...

        Arguments:
            text_file_name (str): name of the text file to load from.
            offset (int): line to start reading from
            shuffle (bool): If True, randomly shuffle loaded data
        """
        text_file_path = utils.download_if_needed(text_file_name)
        text_file = open(text_file_path, "r")
        raw_lines = text_file.readlines()[offset:]
        raw_lines = [self._clean_example(ex) for ex in raw_lines]
        self.examples = [self._process_example_from_file(ex) for ex in raw_lines]
        self._i = 0
        text_file.close()
        if shuffle:
            random.shuffle(self.examples)

    def _clean_example(self, ex):
        """Optionally pre-processes an input string before some tokenization.

        Only necessary for some datasets.
        """
        return ex

    def _load_from_df(self, df, offset=0, shuffle=False, xcol=None, ycol=None):

        """Loads from pandas dataframe
        df : dataframe to load from
        xcol : column to use as x
        ycol : column to use as label
        """

        # if xcol to be used is specified
        new_df = df
        if xcol is not None:
            new_df = pd.DataFrame({xcol: df[xcol]})
        if ycol is not None:
            new_df[ycol] = df[ycol]
        df = new_df
        self.examples = list(df.to_records(index=False))
        self.examples = self.examples[offset:]
        self._i = 0
        if shuffle:
            random.shuffle(self.examples)

    def _load_from_csv(self, path, header=None, sep=",", **kwargs):
        """Loads from csv file"""
        df = pd.read_csv(path, header=header, sep=sep)
        self.examples = self._from_df(df, **kwargs)

    def _load_from_lists(self, list_x_y, offset=0, shuffle=False):
        """Loads from a list
        list_x_y : iterable [(text, label)]
        offset : start
        """
        self.examples = list_x_y
        self.examples = self.examples[offset:]
        self._i = 0
        if shuffle:
            random.shuffle(self.examples)
