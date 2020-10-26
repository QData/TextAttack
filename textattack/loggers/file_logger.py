"""
Attack Logs to file
========================
"""

import os
import sys

import terminaltables

from .logger import Logger


class FileLogger(Logger):
    """Logs the results of an attack to a file, or `stdout`."""

    def __init__(self, filename="", stdout=False):
        self.stdout = stdout
        self.filename = filename
        if stdout:
            self.fout = sys.stdout
        elif isinstance(filename, str):
            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.fout = open(filename, "w")
        else:
            self.fout = filename
        self.num_results = 0

    def __getstate__(self):
        # Temporarily save file handle b/c we can't copy it
        state = {i: self.__dict__[i] for i in self.__dict__ if i != "fout"}
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if self.stdout:
            self.fout = sys.stdout
        else:
            self.fout = open(self.filename, "a")

    def log_attack_result(self, result):
        self.num_results += 1
        color_method = "ansi" if self.stdout else "file"
        self.fout.write(
            "-" * 45 + " Result " + str(self.num_results) + " " + "-" * 45 + "\n"
        )
        self.fout.write(result.__str__(color_method=color_method))
        self.fout.write("\n")

    def log_summary_rows(self, rows, title, window_id):
        if self.stdout:
            table_rows = [[title, ""]] + rows
            table = terminaltables.AsciiTable(table_rows)
            self.fout.write(table.table)
        else:
            for row in rows:
                self.fout.write(f"{row[0]} {row[1]}\n")

    def log_sep(self):
        self.fout.write("-" * 90 + "\n")

    def flush(self):
        self.fout.flush()
