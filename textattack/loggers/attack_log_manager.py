import numpy as np

from textattack.attack_results import FailedAttackResult, SkippedAttackResult

from . import CSVLogger, FileLogger, VisdomLogger, WeightsAndBiasesLogger


def default_attack_metrics():
    from textattack.metrics import (
        AccuracyUnderAttack,
        AttackSuccessRate,
        AverageNumberOfQueries,
        AverageNumberOfWords,
        AveragePerturbedWordPercentage,
        ModelAccuracy,
        TotalAttacks,
        TotalFailedAttacks,
        TotalSkippedAttacks,
        TotalSuccessfulAttacks,
    )

    return [
        TotalAttacks,
        TotalSuccessfulAttacks,
        TotalFailedAttacks,
        TotalSkippedAttacks,
        ModelAccuracy,
        AccuracyUnderAttack,
        AttackSuccessRate,
        AveragePerturbedWordPercentage,
        AverageNumberOfWords,
        AverageNumberOfQueries,
    ]


class AttackLogManager:
    """Logs the results of an attack to all attached loggers."""

    def __init__(self, metrics=[]):
        self.loggers = []
        self.attack_results = []

        self.metrics = metrics
        if not len(self.metrics):
            self.metrics = default_attack_metrics()

    def enable_stdout(self):
        self.loggers.append(FileLogger(stdout=True))

    def enable_visdom(self):
        self.loggers.append(VisdomLogger())

    def enable_wandb(self):
        self.loggers.append(WeightsAndBiasesLogger())

    def add_output_file(self, filename):
        self.loggers.append(FileLogger(filename=filename))

    def add_output_csv(self, filename, color_method):
        self.loggers.append(CSVLogger(filename=filename, color_method=color_method))

    def log_result(self, result):
        """Logs an ``AttackResult`` on each of `self.loggers`."""
        self.attack_results.append(result)
        for logger in self.loggers:
            logger.log_attack_result(result)

    def log_results(self, results):
        """Logs an iterable of ``AttackResult`` objects on each of
        `self.loggers`."""
        for result in results:
            self.log_result(result)
        self.log_metrics()

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
        # TODO: log a more complete set of attack details
        # TODO: call this somewhere, or link to it..?
        attack_detail_rows = [
            ["Attack algorithm:", attack_name],
            ["Model:", model_name],
        ]
        self.log_summary_rows(attack_detail_rows, "Attack Details", "attack_details")

    def log_metrics(self):
        # TODO: restore the histogram thing
        # TODO: create semantic distinction between
        #       `log_summary_rows`
        #           and
        #       `log_metrics`
        # TODO: (maybe) in metrics.py:
        #   develop a level of abstraction
        #   that will automatically support percentages/
        #   averages/etc of different precisions (# decimal
        #   points) and stuff, and also hopefully reduce
        #   repeat computations / reuse things better
        #
        # TODO: choose smarter default metrics for seq2seq models,
        #       regression models, maximization recipe
        #
        # TODO: update tutorials to match `log_metrics` API
        #
        # TODO: tutorial/example of adding a custom metric
        #
        # TODO: Add metrics for edit distance and/or USE similarity
        #
        # TODO: show BLEU score for translation
        # 
        # TODO: add --metrics command-line arg to attack
        #
        metric_table_rows = []
        for metric in self.metrics:
            key = metric.key
            value = str(metric.compute(self.attack_results))
            metric_table_rows.append([key, value])

        self.log_summary_rows(
            metric_table_rows, "Attack Results", "attack_results_summary"
        )
