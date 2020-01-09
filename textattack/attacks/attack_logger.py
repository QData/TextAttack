import difflib
import numpy as np
import os
import torch
import random
import statistics
import time

from textattack.attacks import AttackResult, FailedAttackResult
from textattack.loggers import *

class AttackLogger:
    def __init__(self):
        """ Logs the results of an attack to all attached loggers
        """
        self.loggers = []
        self.results = []
        self.max_words_changed = 0
        self.examples_completed = 0
        self.max_seq_length = 10000
        self.perturbed_word_percentages = []
        self.num_words_changed_until_success = [0] * self.max_seq_length
        self.successful_attacks = 0
        self.failed_attacks = 0
        self.skipped_attacks = 0

    def enable_stdout(self):
        self.loggers.append(FileLogger(stdout=True))

    def enable_visdom(self):
        self.loggers.append(VisdomLogger())

    def add_output_file(self, filename):
        self.loggers.append(FileLogger(filename=filename))

    def add_output_csv(self, filename, plain):
        self.loggers.append(CSVLogger(filename=filename, plain=plain))

    def log_skipped(self, tokenized_text):
        self.skipped_attacks += 1

    def log_result(self, result):
        self.results.append(result)
        self.examples_completed += 1
        for logger in self.loggers:
            logger.log_attack_result(result, self.examples_completed)
        if isinstance(result, FailedAttackResult):
            self.failed_attacks += 1
            return
        self.successful_attacks += 1
        num_words_changed =  len(result.original_text.all_words_diff(result.perturbed_text))
        self.num_words_changed_until_success[num_words_changed-1]+=1
        self.max_words_changed = max(self.max_words_changed,num_words_changed)
        if num_words_changed > 0:
            perturbed_word_percentage = num_words_changed * 100.0 / len(result.original_text.words)
        else:
            perturbed_word_percentage = 0
        self.perturbed_word_percentages.append(perturbed_word_percentage)

    def _log_rows(self, rows, title, window_id):
        for logger in self.loggers:
            logger.log_rows(rows, title, window_id)

    def log_sep(self):
        for logger in self.loggers:
            logger.log_sep()

    def flush(self):
        for logger in self.loggers:
            logger.flush()

    def log_attack_details(self, attack_name, is_black_box, model_name):
        attack_type = 'Black Box' if is_black_box else 'White Box'
        attack_detail_rows = [
            ['Attack algorithm:', attack_name],
            ['Attack type:', attack_type],
            ['Model:', model_name],
        ]
        self._log_rows(attack_detail_rows, 'Attack Details', 'attack_details')

    def _log_num_words_changed(self):
        numbins = max(self.max_words_changed, 10)
        for logger in self.loggers:
            logger.log_hist(self.num_words_changed_until_success[:numbins],
                numbins=numbins, title='Num Words Perturbed', window_id='num_words_perturbed')
    
    def log_summary(self, is_black_box):
        total_attacks = len(self.results)
        # Original classifier success rate on these samples.
        original_accuracy = total_attacks * 100.0 / (total_attacks + self.skipped_attacks) 
        original_accuracy = str(round(original_accuracy, 2)) + '%'
        # New classifier success rate on these samples.
        accuracy_under_attack = (total_attacks - self.successful_attacks) * 100.0 / (total_attacks + self.skipped_attacks)
        accuracy_under_attack = str(round(accuracy_under_attack, 2)) + '%'
        # Attack success rate
        if self.successful_attacks + self.failed_attacks == 0:
            attack_success_rate = 0
            average_perc_words_perturbed = 0
        
        else:
            attack_success_rate = self.successful_attacks * 100.0 / (self.successful_attacks + self.failed_attacks) 
            average_perc_words_perturbed = statistics.mean(self.perturbed_word_percentages)
            
        attack_success_rate = str(round(attack_success_rate, 2)) + '%'
        average_perc_words_perturbed = str(round(average_perc_words_perturbed, 2)) + '%'
        
        summary_table_rows = [
            ['Number of successful attacks:', str(self.successful_attacks)],
            ['Number of failed attacks:', str(self.failed_attacks)],
            ['Number of skipped attacks:', str(self.skipped_attacks)],
            ['Original accuracy:', original_accuracy],
            ['Accuracy under attack:', accuracy_under_attack],
            ['Attack success rate:', attack_success_rate],
            ['Average perturbed word %:', average_perc_words_perturbed],
        ]
        
        if is_black_box:
            num_queries = [r.num_queries for r in self.results]
            avg_num_queries = statistics.mean(num_queries) if len(num_queries) else 0
            avg_num_queries = str(round(avg_num_queries, 2))
            summary_table_rows.append(['Avg num queries:', avg_num_queries])
        summary_table_rows.append(['Time to run attack:', f'{time.time() - self.start_time}s'])
        self._log_rows(summary_table_rows, 'Attack Results Summary', 'attack_results_summary')
        self._log_num_words_changed()
