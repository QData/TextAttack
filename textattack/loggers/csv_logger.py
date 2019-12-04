import sys
import os
import pandas as pd
import csv
from textattack.loggers import Logger

class CSVLogger(Logger):
    def __init__(self, filename='results.csv', plain=False):
        self.filename = filename
        self.plain = plain
        self.df = pd.DataFrame()

    def log_attack_result(self, result, examples_completed):
        color_method = None if self.plain else 'file'
        s1, s2 = result.diff_color(color_method)
        row = {'passage_1': s1, 'passage_2': s2} 
        self.df = self.df.append(row, ignore_index=True)

    def flush(self):
        self.df.to_csv(self.filename, quoting=csv.QUOTE_NONNUMERIC)
