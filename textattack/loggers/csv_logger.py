import csv

import pandas as pd

from textattack.shared import AttackedText, logger

from .file_logger import FileLogger
from .logger import Logger


class CSVLogger(Logger):
    """Logs attack results to a CSV."""

    def __init__(self, filename="results.csv", color_method="file"):
        self.filename = filename
        self.color_method = color_method
        self.df = pd.DataFrame()
        self._printed_header = False
        self._file_logger = FileLogger(filename=filename, stdout=False)

    def log_attack_result(self, result):
        original_text, perturbed_text = result.diff_color(self.color_method)
        original_text = original_text.replace("\n", AttackedText.SPLIT_TOKEN)
        perturbed_text = perturbed_text.replace("\n", AttackedText.SPLIT_TOKEN)
        result_type = result.__class__.__name__.replace("AttackResult", "")
        row = {
            "original_text": original_text,
            "perturbed_text": perturbed_text,
            "original_score": result.original_result.score,
            "perturbed_score": result.perturbed_result.score,
            "original_output": result.original_result.output,
            "perturbed_output": result.perturbed_result.output,
            "ground_truth_output": result.original_result.ground_truth_output,
            "num_queries": result.num_queries,
            "result_type": result_type,
        }
        self.df = self.df.append(row, ignore_index=True)
        df_str = self.df.to_csv().strip()
        if self._printed_header:
            last_line_pos = df_str.rfind("\n") + 1
            df_last_line_str = df_str[last_line_pos:]
            self._file_logger.fout.write(df_last_line_str)
        else:
            self._file_logger.fout.write(df_str)
        self._file_logger.fout.write("\n")
