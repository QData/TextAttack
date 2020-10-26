"""
Attack Logger Wrapper
========================
"""


from abc import ABC


class Logger(ABC):
    """An abstract class for different methods of logging attack results."""

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
