"""
Managing Attack Logs.
========================
"""

import numpy as np

from textattack.attack_metrics import AttackSuccessRate, WordsPerturbed, AttackQueries

from . import CSVLogger, FileLogger, VisdomLogger, WeightsAndBiasesLogger


class AttackLogManager:
    """Logs the results of an attack to all attached loggers."""

    def __init__(self):
        self.loggers = []
        self.results = []

    def enable_stdout(self):
        self.loggers.append(FileLogger(stdout=True))

    def enable_visdom(self):
        self.loggers.append(VisdomLogger())

    def enable_wandb(self):
        self.loggers.append(WeightsAndBiasesLogger())

    def disable_color(self):
        self.loggers.append(FileLogger(stdout=True, color_method="file"))

    def add_output_file(self, filename, color_method):
        self.loggers.append(FileLogger(filename=filename, color_method=color_method))

    def add_output_csv(self, filename, color_method):
        self.loggers.append(CSVLogger(filename=filename, color_method=color_method))

    def log_result(self, result):
        """Logs an ``AttackResult`` on each of `self.loggers`."""
        self.results.append(result)
        for logger in self.loggers:
            logger.log_attack_result(result)

    def log_results(self, results):
        """Logs an iterable of ``AttackResult`` objects on each of
        `self.loggers`."""
        for result in results:
            self.log_result(result)
        self.log_summary()

    def log_summary_rows(self, rows, title, window_id):
        for logger in self.loggers:
            logger.log_summary_rows(rows, title, window_id)

    def log_sep(self):
        for logger in self.loggers:
            logger.log_sep()

    def flush(self):
        for logger in self.loggers:
            logger.flush()

    def log_attack_details(self, attack_name, model_name):
        # @TODO log a more complete set of attack details
        attack_detail_rows = [
            ["Attack algorithm:", attack_name],
            ["Model:", model_name],
        ]
        self.log_summary_rows(attack_detail_rows, "Attack Details", "attack_details")

    def log_summary(self):
        total_attacks = len(self.results)
        if total_attacks == 0:
            return

        # Default metrics - calculated on every attack
        attack_success_stats = AttackSuccessRate(self.results)
        words_perturbed_stats = WordsPerturbed(self.results)
        attack_query_stats = AttackQueries(self.results)

        # @TODO generate this table based on user input - each column in specific class
        # Example to demonstrate:
        # summary_table_rows = attack_success_stats.display_row() + words_perturbed_stats.display_row() + ...
        summary_table_rows = [
            [
                "Number of successful attacks:",
                attack_success_stats.successful_attacks_num(),
            ],
            ["Number of failed attacks:", attack_success_stats.failed_attacks_num()],
            ["Number of skipped attacks:", attack_success_stats.skipped_attacks_num()],
            [
                "Original accuracy:",
                str(attack_success_stats.original_accuracy_perc()) + "%",
            ],
            [
                "Accuracy under attack:",
                str(attack_success_stats.attack_accuracy_perc()) + "%",
            ],
            [
                "Attack success rate:",
                str(attack_success_stats.attack_success_rate_perc()) + "%",
            ],
            [
                "Average perturbed word %:",
                str(words_perturbed_stats.avg_number_word_perturbed_num()) + "%",
            ],
            [
                "Average num. words per input:",
                words_perturbed_stats.avg_perturbation_perc(),
            ],
        ]

        summary_table_rows.append(
            ["Avg num queries:", attack_query_stats.avg_num_queries_num()]
        )
        self.log_summary_rows(
            summary_table_rows, "Attack Results", "attack_results_summary"
        )
        # Show histogram of words changed.
        numbins = max(words_perturbed_stats.max_words_changed_num(), 10)
        for logger in self.loggers:
            logger.log_hist(
                words_perturbed_stats.num_words_changed_until_success_num()[:numbins],
                numbins=numbins,
                title="Num Words Perturbed",
                window_id="num_words_perturbed",
            )
