# @TODO all files should have some copyright header or something

"""" @TODOs:
    - Example of using with Pytorch and Tensorflow model.
    - sphinx documentation
    - add recipes with state-of-the-art attacks
    - add unit tests
    - add pep8 standard
    - upload sample models and datasets
    - add logger... we should never call print()
    - make it much quieter when we load pretrained BERT. It's so noisy right now :(
    - try to refer to 'text' not 'sentences' (better terminology)
    - make this into a pip package (not on pypi, just a local package)
"""

import difflib
import torch
import utils
import random
import statistics

from tokenized_text import TokenizedText
from loggers import VisdomLogger

class Attack:
    """ An attack generates adversarial examples on text. """
    def __init__(self, model, transformation, attack_name):
        """ Initialize an attack object.
        
        Attacks can be run multiple times
        
         @TODO should `tokenizer` be an additional parameter or should
            we assume every model has a .tokenizer ?
        """
        self.model = model
        self.transformation = transformation
        self.attack_name = attack_name
        # List of files to output to.
        self.output_files = []
        self.output_to_terminal = True
        self.output_to_visdom = False
        self.visdom = VisdomLogger()
        
    def enable_visdom(self):
        """ When attack runs, it will send statistics to Visdom """
        self.output_to_visdom = True
    
    def add_output_file(self, file):
        """ When attack runs, it will output to this file. """
        if isinstance(file, str):
            file = open(file, 'w')
        self.output_files.append(file)
    
    def _attack_one(self, label, tokenized_text):
        """ Perturbs `text` to until `self.model` gives a different label
            than `label`. """
        raise NotImplementedError()
    
    def _call_model(self, tokenized_text_list):
        """ Returns model predictions for a list of TokenizedText objects. """
        # @todo support models that take text instead of IDs.
        ids = torch.tensor([t.ids for t in tokenized_text_list])
        ids = ids.to(utils.get_device())
        return self.model(ids).squeeze()
    
    def attack(self, dataset, shuffle=False):
        """ Runs an attack on some data and outputs results.
        
            - dataset: an iterable of (label, text) pairs
        """
        if shuffle:
            random.shuffle(dataset)
        
        results = []
        for label, text in dataset:
            tokenized_text = TokenizedText(self.model, text)
            result = self._attack_one(label, tokenized_text)
            results.append(result)
        
        # @TODO Support failed attacks. Right now they'll throw an error
        
        if self.output_to_terminal:
            for i, result in enumerate(results):
                print('-'*35, 'Result', str(i+1), '-'*35)
                result.print_()
                print()
        
        if self.output_files:
            for output_file in self.output_files:
                for result in results:
                    output_file.write(str(result) + '\n')
        
        if self.output_to_visdom:
            # @TODO Support logging to Visdom.
            sample_rows = []
            num_words_changed_until_success = [0] * (self.max_depth+5)
            perturbed_word_percentages = []
            input_text_tkns = result.original_text.text.split()
            output_text_tkns = result.perturbed_text.text.split()
            for result in results:
                row = []
                labelchange = str(result.original_label)+" -> "+str(result.perturbed_label)
                row.append(labelchange)
                text1, text2, num_words_changed = result.diff(html=True)
                row.append(text1)
                row.append(text2)
                num_words_changed_until_success[num_words_changed-1]+=1
                if num_words_changed > 0:
                    perturbed_word_percentage = num_words_changed * 100.0 / len(input_text_tkns)
                    perturbed_word_percentages.append(perturbed_word_percentage)
                else:
                    perturbed_word_percentage = 0
                perturbed_word_percentage_str = str(round(perturbed_word_percentage, 2))
                sample_rows.append(row)
            self.log_samples(sample_rows)
            
            self.log_num_words_changed(num_words_changed_until_success)
            
            attack_detail_rows = [
                ['Attack algorithm:', self.attack_name],
                ['Model:', self.model.name],
                ['Word suggester:', self.transformation.name],
                ['Content preservation method:', self.transformation.constraints[0].metric],
                ['Content preservation threshold:', self.transformation.constraints[0].threshold],
            ]
            self.log_attack_details(attack_detail_rows)
            
            total_attacks = len(results)
            num_failed_attacks = num_words_changed_until_success[-1]
            num_successful_attacks = total_attacks - num_failed_attacks
            num_already_misclassified_samples = num_words_changed_until_success[0]
            # Original classifier success rate on these samples.
            original_accuracy = (total_attacks - num_already_misclassified_samples) * 100.0 / total_attacks
            original_accuracy = str(round(original_accuracy, 2)) + '%'
            # New classifier success rate on these samples.
            attack_accuracy = (total_attacks - num_successful_attacks - num_already_misclassified_samples) * 100.0 / total_attacks
            attack_accuracy = str(round(attack_accuracy, 2)) + '%'
            # Average % of words perturbed per sample.
            average_perc_words_perturbed = statistics.mean(perturbed_word_percentages)
            average_perc_words_perturbed = str(round(average_perc_words_perturbed, 2)) + '%'
            summary_table_rows = [
                ['Total number of attacks:', total_attacks],
                ['Number of failed attacks:', num_failed_attacks],
                ['Original accuracy:', original_accuracy],
                ['Attack accuracy:', attack_accuracy],
                ['Perturbed word percentage:', average_perc_words_perturbed],
            ]
            self.log_summary(summary_table_rows)
            
        print('-'*80)
        
        return results
    
    def log_samples(self, rows):
        self.visdom.table(rows, window_id="results", title="Attack Results")
        
    def log_attack_details(self, rows):
        self.visdom.table(rows, title='Attack Details',
                    window_id='attack_details')
        
    def log_summary(self, rows):
        self.visdom.table(rows, title='Summary',
                window_id='summary_table')

    def log_num_words_changed(self, num_words_changed):
        numbins = len(num_words_changed)
        #self.visdom.hist(num_words_changed,
        #    numbins=numbins, title='Result', window_id='powers_hist')
        self.visdom.bar(num_words_changed,
            numbins=numbins, title='Result', window_id='powers_hist')

class AttackResult:
    """ Result of an Attack run on a single (label, text_input) pair. 
    
        @TODO support attacks that fail (no perturbed label/text)
    """
    def __init__(self, original_text, perturbed_text, original_label,
        perturbed_label):
        self.original_text = original_text
        self.perturbed_text = perturbed_text
        self.original_label = original_label
        self.perturbed_label = perturbed_label
    
    def __data__(self):
        data = (self.original_text, self.original_label, self.perturbed_text,
                self.perturbed_label)
        return tuple(map(str, data))
    
    def __str__(self):
        return '\n'.join(self.__data__())
    
    def diff(self, html=False):
        """ Shows the difference between two strings in color.
        
        @TODO abstract to work for general paraphrase.
        """
        #@TODO: Support printing to HTML in some cases.
        if html:
            _color = utils.color_text_html
        else:
            _color = utils.color_text_terminal
        _diff = utils.diff_indices
        t1 = self.original_text
        t2 = self.perturbed_text
        words1 = t1.words()
        words2 = t2.words()
        
        indices = _diff(words1,words2)
        
        c1 = utils.color_from_label(self.original_label)
        c2 = utils.color_from_label(self.perturbed_label)
        
        new_w1s = []
        new_w2s = []
        
        for i in indices:
            w1 = words1[i]
            w2 = words2[i]
            new_w1s.append(_color(w1, c1))
            new_w2s.append(_color(w2, c2))
        
        t1 = self.original_text.replace_words_at_indices(indices, new_w1s)
        t2 = self.original_text.replace_words_at_indices(indices, new_w2s)
                
        return (str(t1), str(t2), len(indices))
    
    def print_(self):
        print(str(self.original_label), '-->', str(self.perturbed_label))
        print('\n'.join(self.diff()[:2]))

if __name__ == '__main__':
    import attacks
    import constraints
    from datasets import YelpSentiment
    from models import BertForSentimentClassification
    from transformations import WordSwapEmbedding
    
    # @TODO: Running attack.py should parse args and run script-based attacks 
    #       (as opposed to code-based attacks)
    model = BertForSentimentClassification()
    
    transformation = WordSwapEmbedding()
    
    transformation.add_constraints((
        # constraints.syntax.LanguageTool(1),
        constraints.semantics.UniversalSentenceEncoder(0.9, metric='cosine'),
        )
    )
    
    attack = attacks.GreedyWordSwap(model, transformation)
    
    yelp_data = YelpSentiment(n=1)
    # yelp_data = [
    #     (1, 'I hate this Restaurant!'), 
    #     (0, "Texas Jack's has amazing food.")
    # ]
    
    attack.enable_visdom()
    attack.add_output_file(open('outputs/test.txt', 'w'))
    import sys
    attack.add_output_file(sys.stdout)
    attack.attack(yelp_data, shuffle=False)