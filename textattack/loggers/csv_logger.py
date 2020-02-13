import sys
import os
import pandas as pd
import csv

from textattack.attack_results import FailedAttackResult
from textattack.shared.utils import get_logger
from .logger import Logger

class CSVLogger(Logger):
    def __init__(self, filename='results.csv', plain=False):
        self.filename = filename
        self.plain = plain
        self.df = pd.DataFrame()
        self._flushed = True

    def log_attack_result(self, result):
        if isinstance(result, FailedAttackResult):
            return
        color_method = None if self.plain else 'file'
        s1, s2 = result.diff_color(color_method)
        row = {'passage_1': s1, 'passage_2': s2, 'score_1': result.orig_score, 'score_2': result.perturbed_score, 'label_1': int(result.original_label), 'label_2': int(result.perturbed_label)} 
        self.df = self.df.append(row, ignore_index=True)
        self._flushed = False

    def flush(self):
        self.df.to_csv(self.filename, quoting=csv.QUOTE_NONNUMERIC, index=False)
        self._flushed = True
    
    def __del__(self):
        if not self._flushed:
            get_logger().warning('CSVLogger exiting without calling flush().')
