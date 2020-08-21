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

    def log_sep(self):
        for logger in self.loggers:
            logger.log_sep()

    def flush(self):
        for logger in self.loggers:
            logger.flush()

    def log_metrics(self):
        #
        # TODO: ask Eli if metrics are properly computed when results are MaximizedAttackResults
        #
        # TODO: restore the histogram thing
        #
        # TODO: choose smarter default metrics for seq2seq models,
        #       regression models, maximization recipe
        #       (show BLEU score for translation)
        #
        # TODO: update tutorials to match `log_metrics` API
        #
        # TODO: tutorial/example of adding a custom metric
        #
        metric_table_rows = []
        for metric in self.metrics:
            key = metric.key
            value = metric.compute_str(self.attack_results)
            metric_table_rows.append([key, value])

        # Print metrics to `self.loggers`.
        for logger in self.loggers:
            logger.log_summary_rows(rows, "Attack Results", "attack_results_summary")
