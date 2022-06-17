"""
Attack Summary Results Logs to Json
========================
"""

import json

from textattack.shared import logger

from .logger import Logger


class JsonSummaryLogger(Logger):
    def __init__(self, filename="results_summary.json"):
        logger.info(f"Logging Summary to JSON at path {filename}")
        self.filename = filename
        self.json_dictionary = {}
        self._flushed = True

    def log_summary_rows(self, rows, title, window_id):
        self.json_dictionary[title] = {}
        for i in range(len(rows)):
            row = rows[i]
            if isinstance(row[1], str):
                try:
                    row[1] = row[1].replace("%", "")
                    row[1] = float(row[1])
                except ValueError:
                    raise ValueError(
                        f'Unable to convert row value "{row[1]}" for Attack Result "{row[0]}" into float'
                    )

        for metric, summary in rows:
            self.json_dictionary[title][metric] = summary

        self._flushed = False

    def flush(self):
        with open(self.filename, "w") as f:
            json.dump(self.json_dictionary, f, indent=4)

        self._flushed = True

    def close(self):
        super().close()
