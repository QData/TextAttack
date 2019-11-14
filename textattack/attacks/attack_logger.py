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
        from textattack.attacks import Attack, AttackResult, FailedAttackResult
        self.visdom = VisdomLogger()
        self.attack = attack
        self.results = None
        self.num_words_changed_until_success = []
        self.perturbed_word_percentages = []

    def log_samples(self, results):
        self.results = results
        sample_rows = []
        self.num_words_changed_until_success = [0] * (self.attack.max_depth+5)
        self.perturbed_word_percentages = []
        for result in self.results:
            input_text_tkns = result.original_text.text.split()
            output_text_tkns = result.perturbed_text.text.split()
            row = []
            labelchange = str(result.original_label)+" -> "+str(result.perturbed_label)
            row.append(labelchange)
            text1, text2, num_words_changed = self.diff(result, html=True)
            row.append(text1)
            row.append(text2)
            self.num_words_changed_until_success[num_words_changed]+=1
            if num_words_changed > 0:
                if num_words_changed > len(input_text_tkns):
                    num_words_changed = len(input_text_tkns)
                perturbed_word_percentage = num_words_changed * 100.0 / len(input_text_tkns)
            else:
                perturbed_word_percentage = 0
            self.perturbed_word_percentages.append(perturbed_word_percentage)
            perturbed_word_percentage_str = str(round(perturbed_word_percentage, 2))
            sample_rows.append(row)
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
        
        if indices == []:
            indices = range(self.attack.max_depth+4)
        
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
                
        return (str(t1), str(t2), len(indices))
        
    def diff_indices(self, words1, words2):
        indices = []
        for i in range(min(len(words1), len(words2))):
            w1 = words1[i]
            w2 = words2[i]
            if w1 != w2:
                indices.append(i)
        return indices
            
    def log_num_words_changed(self):        
        numbins = len(self.num_words_changed_until_success)
        hist = [0] * (self.attack.max_depth+5)
        for i in range(len(self.num_words_changed_until_success)):
            if i == 0:
                continue
            hist[i-1] = self.num_words_changed_until_success[i]
            
        self.visdom.bar(hist,
            numbins=numbins, title='Result', window_id='powers_hist')
            
    def log_attack_details(self):
        attack_detail_rows = [
            ['Attack algorithm:', str(self.attack)],
            ['Model:', str(self.attack.model)],
            ['Word suggester:', str(self.attack.transformation)],
        ]
        self.visdom.table(attack_detail_rows, title='Attack Details',
                    window_id='attack_details')
    
    def log_summary(self):
        print(self.num_words_changed_until_success)
        total_attacks = len(self.results)
        num_failed_attacks = self.num_words_changed_until_success[-1]
        num_successful_attacks = total_attacks - num_failed_attacks
        num_already_misclassified_samples = self.num_words_changed_until_success[0]
        # Original classifier success rate on these samples.
        original_accuracy = (total_attacks - num_already_misclassified_samples) * 100.0 / total_attacks
        original_accuracy = str(round(original_accuracy, 2)) + '%'
        # New classifier success rate on these samples.
        attack_accuracy = (total_attacks - num_successful_attacks - num_already_misclassified_samples) * 100.0 / total_attacks
        attack_accuracy = str(round(attack_accuracy, 2)) + '%'
        # Average % of words perturbed per sample.
        average_perc_words_perturbed = statistics.mean(self.perturbed_word_percentages)
        average_perc_words_perturbed = str(round(average_perc_words_perturbed, 2)) + '%'
        summary_table_rows = [
            ['Total number of attacks:', total_attacks],
            ['Number of failed attacks:', num_failed_attacks],
            ['Original accuracy:', original_accuracy],
            ['Attack accuracy:', attack_accuracy],
            ['Perturbed word percentage:', average_perc_words_perturbed],
        ]
        self.visdom.table(summary_table_rows, title='Summary',
                window_id='summary_table')
