import csv
import os
import sys

import pandas as pd

from textattack.attack_results import FailedAttackResult
from textattack.shared import logger

from .logger import Logger


class CSVLogger(Logger):
    """ Logs attack results to a CSV. """

    def __init__(self, filename="results.csv", color_method="file"):
        self.filename = filename
        self.color_method = color_method
        self.df = pd.DataFrame()
        self._flushed = True

    def log_attack_result(self, result):
        if isinstance(result, FailedAttackResult):
            return
        original_text, perturbed_text = result.diff_color(self.color_method)
        row = {
            "original_text": original_text,
            "perturbed_text": perturbed_text,
            "original_score": result.original_result.score,
            "perturbed_score": result.perturbed_result.score,
            "original_output": result.original_result.output,
            "perturbed_output": result.perturbed_result.output,
            "num_queries": result.num_queries,
        }
        self.df = self.df.append(row, ignore_index=True)
        self._flushed = False

    def flush(self):
        self.df.to_csv(self.filename, quoting=csv.QUOTE_NONNUMERIC, index=False)
        self._flushed = True

    def __del__(self):
        if not self._flushed:
            logger.warning("CSVLogger exiting without calling flush().")
