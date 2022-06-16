"""
Attack Summary Results Logs to Json
========================
"""

import os
import sys

from textattack.shared import logger

from .logger import Logger


class JsonSummaryLogger(Logger):

    def __init__(self):
        pass

    def log_attack_result(self, result, examples_completed):
        pass

    def log_summary_rows(self, rows, title, window_id):
        pass

    def log_hist(self, arr, numbins, title, window_id):
        pass

    def log_sep(self):
        pass

    def flush(self):
        pass

    def close(self):
        pass

