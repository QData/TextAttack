"""
Managing Attack Logs.
========================
"""

from typing import Dict, Optional

from textattack.metrics.attack_metrics import (
    AttackQueries,
    AttackSuccessRate,
    WordsPerturbed,
)
from textattack.metrics.quality_metrics import Perplexity, USEMetric

from . import (
    CSVLogger,
    FileLogger,
    JsonSummaryLogger,
    VisdomLogger,
    WeightsAndBiasesLogger,
)


class AttackLogManager:
    """Logs the results of an attack to all attached loggers."""

    # metrics maps strings (metric names) to textattack.metric.Metric objects
    metrics: Dict

    def __init__(self, metrics: Optional[Dict]):
        self.loggers = []
        self.results = []
        self.enable_advance_metrics = False
        if metrics is None:
            self.metrics = {}
        else:
            self.metrics = metrics

    def enable_stdout(self):
        self.loggers.append(FileLogger(stdout=True))

    def enable_visdom(self):
        self.loggers.append(VisdomLogger())

    def enable_wandb(self, **kwargs):
        self.loggers.append(WeightsAndBiasesLogger(**kwargs))

    def disable_color(self):
        self.loggers.append(FileLogger(stdout=True, color_method="file"))

    def add_output_file(self, filename, color_method):
        self.loggers.append(FileLogger(filename=filename, color_method=color_method))

    def add_output_csv(self, filename, color_method):
        self.loggers.append(CSVLogger(filename=filename, color_method=color_method))

    def add_output_summary_json(self, filename):
        self.loggers.append(JsonSummaryLogger(filename=filename))

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
        attack_success_stats = AttackSuccessRate().calculate(self.results)
        words_perturbed_stats = WordsPerturbed().calculate(self.results)
        attack_query_stats = AttackQueries().calculate(self.results)

        # @TODO generate this table based on user input - each column in specific class
        # Example to demonstrate:
        # summary_table_rows = attack_success_stats.display_row() + words_perturbed_stats.display_row() + ...
        summary_table_rows = [
            [
                "Number of successful attacks:",
                attack_success_stats["successful_attacks"],
            ],
            ["Number of failed attacks:", attack_success_stats["failed_attacks"]],
            ["Number of skipped attacks:", attack_success_stats["skipped_attacks"]],
            [
                "Original accuracy:",
                str(attack_success_stats["original_accuracy"]) + "%",
            ],
            [
                "Accuracy under attack:",
                str(attack_success_stats["attack_accuracy_perc"]) + "%",
            ],
            [
                "Attack success rate:",
                str(attack_success_stats["attack_success_rate"]) + "%",
            ],
            [
                "Average perturbed word %:",
                str(words_perturbed_stats["avg_word_perturbed_perc"]) + "%",
            ],
            [
                "Average num. words per input:",
                words_perturbed_stats["avg_word_perturbed"],
            ],
        ]

        summary_table_rows.append(
            ["Avg num queries:", attack_query_stats["avg_num_queries"]]
        )

        for metric_name, metric in self.metrics.items():
            summary_table_rows.append([metric_name, metric.calculate(self.results)])

        if self.enable_advance_metrics:
            perplexity_stats = Perplexity().calculate(self.results)
            use_stats = USEMetric().calculate(self.results)

            summary_table_rows.append(
                [
                    "Average Original Perplexity:",
                    perplexity_stats["avg_original_perplexity"],
                ]
            )

            summary_table_rows.append(
                [
                    "Average Attack Perplexity:",
                    perplexity_stats["avg_attack_perplexity"],
                ]
            )
            summary_table_rows.append(
                ["Average Attack USE Score:", use_stats["avg_attack_use_score"]]
            )

        self.log_summary_rows(
            summary_table_rows, "Attack Results", "attack_results_summary"
        )
        # Show histogram of words changed.
        numbins = max(words_perturbed_stats["max_words_changed"], 10)
        for logger in self.loggers:
            logger.log_hist(
                words_perturbed_stats["num_words_changed_until_success"][:numbins],
                numbins=numbins,
                title="Num Words Perturbed",
                window_id="num_words_perturbed",
            )
