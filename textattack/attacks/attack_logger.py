import difflib
import numpy as np
import os
import torch
import random
import statistics

from textattack import utils as utils

from textattack.constraints import Constraint
from textattack.tokenized_text import TokenizedText
from textattack.loggers import VisdomLogger

class AttackLogger:
    def __init__(self, attack):
        """ Logs the results of an attack
        """
        from textattack.attacks import Attack, AttackResult, FailedAttackResult
        self.attack = attack
        self.attack_type = 'White Box' if self.attack.model_description == str(self.attack.model) else 'Black Box'
        self.results = None
        self.num_words_changed_until_success = []
        self.perturbed_word_percentages = []
        self.max_words_changed = 0
        self.examples_completed = 0
        self.max_seq_length = 10000

    def enable_visdom(self):
        self.visdom = VisdomLogger()

    def log_result(self, result):
        self.examples_completed += 1
        if self.attack.output_to_stdout:
            print('-'*35, 'Result', str(self.examples_completed), '-'*35)
            result.print_()
            print()

    def _log_rows_to_files_and_visdom(self, rows, title, window_id):
        for output_file in self.attack.output_files:
            for row in rows:
                output_file.write(row[0] + ' ' + row[1] + '\n')
        if self.attack.output_to_visdom:
            self.visdom.table(rows, title=title, window_id=window_id)

    def log_samples(self, results):
        for output_file in self.attack.output_files:
            for result in results:
                output_file.write(str(result) + '\n')
        self.results = results
        sample_rows = []
        self.num_words_changed_until_success = [0] * self.max_seq_length
        self.perturbed_word_percentages = []
        for result in self.results:
            if result.original_text.text == result.perturbed_text.text:
                continue
            num_words = len(result.original_text.words())
            row = []
            labelchange = str(result.original_label)+" -> "+str(result.perturbed_label)
            row.append(labelchange)
            text1, text2, num_words_changed = self.diff(result, html=True)
            row.append(text1)
            row.append(text2)
            self.num_words_changed_until_success[num_words_changed-1]+=1
            self.max_words_changed = max(self.max_words_changed,num_words_changed)
            if num_words_changed > 0:
                perturbed_word_percentage = num_words_changed * 100.0 / num_words
            else:
                perturbed_word_percentage = 0
            self.perturbed_word_percentages.append(perturbed_word_percentage)
            sample_rows.append(row)
        if self.attack.output_to_visdom:
            self.visdom.table(sample_rows, window_id="results", title="Attack Results")
            
    def diff(self, result, html=False):
        """ Shows the difference between two strings in color.
        
        @TODO abstract to work for general paraphrase.
        """
        #@TODO: Support printing to HTML in some cases.
        if html:
            _color = utils.color_text_html
        else:
            _color = utils.color_text_terminal
            
        t1 = result.original_text
        t2 = result.perturbed_text
            
        words1 = t1.words()
        words2 = t2.words()
        
        indices = self.diff_indices(words1,words2)
        
        c1 = utils.color_from_label(result.original_label)
        c2 = utils.color_from_label(result.perturbed_label)
        
        new_w1s = []
        new_w2s = []
        r_indices = []
        
        for i in indices:
            if i<len(words1):
                r_indices.append(i)
                w1 = words1[i]
                w2 = words2[i]
                new_w1s.append(_color(w1, c1))
                new_w2s.append(_color(w2, c2))
        
        t1 = result.original_text.replace_words_at_indices(r_indices, new_w1s)
        t2 = result.original_text.replace_words_at_indices(r_indices, new_w2s)
                
        return (t1.text, t2.text, len(indices))
        
    def diff_indices(self, words1, words2):
        indices = []
        for i in range(min(len(words1), len(words2))):
            w1 = words1[i]
            w2 = words2[i]
            if w1 != w2:
                indices.append(i)
        return indices
            
    def log_num_words_changed(self):        
        if not self.attack.output_to_visdom:
            return
        numbins = max(self.max_words_changed,10)
            
        self.visdom.bar(self.num_words_changed_until_success[:numbins],
            numbins=numbins, title='Num Words Perturbed', window_id='powers_hist')
            
    def log_attack_details(self):
        attack_detail_rows = [
            ['Attack algorithm:', self.attack.__class__.__name__],
            ['Attack type:', self.attack_type],
            ['Model:', self.attack.model_description],
        ]
        self._log_rows_to_files_and_visdom(attack_detail_rows, 'Attack Details', 'attack_details')
    
    def log_summary(self):
        total_attacks = len(self.results)
        # Original classifier success rate on these samples.
        original_accuracy = total_attacks * 100.0 / (total_attacks + self.attack.skipped_attacks) 
        original_accuracy = str(round(original_accuracy, 2)) + '%'
        # New classifier success rate on these samples.
        accuracy_under_attack = (total_attacks - self.attack.successful_attacks) * 100.0 / (total_attacks + self.attack.skipped_attacks)
        accuracy_under_attack = str(round(accuracy_under_attack, 2)) + '%'
        # Attack success rate
        attack_success_rate = self.attack.successful_attacks * 100.0 / (self.attack.successful_attacks + self.attack.failed_attacks) 
        attack_success_rate = str(round(attack_success_rate, 2)) + '%'
        # Average % of words perturbed per sample.
        average_perc_words_perturbed = statistics.mean(self.perturbed_word_percentages)
        average_perc_words_perturbed = str(round(average_perc_words_perturbed, 2)) + '%'
        summary_table_rows = [
            ['Number of samples attacked:', str(total_attacks)],
            ['Number of failed attacks:', str(self.attack.failed_attacks)],
            ['Original accuracy:', original_accuracy],
            ['Accuracy under attack:', accuracy_under_attack],
            ['Attack success rate:', attack_success_rate],
            ['Average perturbed word %:', average_perc_words_perturbed],
        ]
        if self.attack_type == 'Black Box':
            num_queries = [r.num_queries for r in self.results]
            avg_num_queries = statistics.mean(num_queries)
            avg_num_queries = str(round(avg_num_queries, 2))
            summary_table_rows.append(['Avg num queries:', avg_num_queries])
        self._log_rows_to_files_and_visdom(summary_table_rows, 'Summary', 'summary_table')
