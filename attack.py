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
"""

import difflib
import torch
import utils

from tokenized_text import TokenizedText

class Attack:
    
    def __init__(self, model, perturbation):
        """ Initialize an attack object.
        
        Attacks can be run multiple times
        
         @TODO should `tokenizer` be an additional parameter or should
            we assume every model has a .tokenizer ?
        """
        self.model = model
        self.perturbation = perturbation
        # List of files to output to.
        self.output_files = []
        self.output_to_terminal = True
    
    def add_constraint(self, constraint):
        """ Add constraint to attack. """
        raise NotImplementedException()
    
    def add_constraints(self, constraints):
        """ Add multiple constraints.
        """
        for constraint in constraints:
            self.add_constraint(constraint)
    
    def add_output_file(self, file):
        """ When attack runs, it will output to this file. """
        if isinstance(file, str):
            file = open(file, 'w')
        self.output_files.append(file)
    
    def _attack_one(self, label, tokenized_text):
        """ Perturbs `text` to until `self.model` gives a different label
            than `label`. """
        raise NotImplementedException()
    
    def _call_model(self, tokenized_text_list):
        """ Returns model predictions for a list of TokenizedText objects. """
        # @todo support models that take text instead of IDs.
        ids = torch.tensor([t.ids for t in tokenized_text_list])
        ids = ids.to(utils.get_device())
        return self.model(ids).squeeze()
    
    def attack(self, dataset, n=None, shuffle=False):
        """ Runs an attack on some data and outputs results.
        
            - dataset: an iterable of (label, text) pairs
        """
        if shuffle:
            random.shuffle(dataset)

        _i = 0
        results = []
        for label, text in dataset:
            tokenized_text = TokenizedText(self.model, text)
            result = self._attack_one(label, tokenized_text)
            results.append(result)
            _i += 1
            if n and _i > n:
                break
        
        for output_file in self.output_files:
            for result in results:
                output_file.write(str(result) + '\n')
        
        if self.output_to_terminal:
            for i, result in enumerate(results):
                print('-'*35, 'Result', str(i+1), '-'*35)
                result.print_()
                print()
        print('-'*80)
        
        return results

class AttackResult:
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
    
    def diff(self):
        """ Shows the difference between two strings in color.
        
        @TODO abstract to work for general paraphrase.
        """
        #@TODO: Support printing to HTML in some cases.
        _color = utils.color_text_terminal
        t1 = self.original_text
        t2 = self.perturbed_text
        
        words1 = t1.words()
        words2 = t2.words()
        
        c1 = utils.color_from_label(self.original_label)
        c2 = utils.color_from_label(self.perturbed_label)
        new_is = []
        new_w1s = []
        new_w2s = []
        for i in range(min(len(words1), len(words2))):
            w1 = words1[i]
            w2 = words2[i]
            if w1 != w2:
                new_is.append(i)
                new_w1s.append(_color(w1, c1))
                new_w2s.append(_color(w2, c2))
        
        t1 = self.original_text.replace_words_at_indices(new_is, new_w1s)
        t2 = self.original_text.replace_words_at_indices(new_is, new_w2s)
                
        return (str(t1), str(t2))
    
    def print_(self):
        print(str(self.original_label), '-->', str(self.perturbed_label))
        print('\n'.join(self.diff()))

if __name__ == '__main__':
    import attacks
    from models import BertForSentimentClassification
    from perturbations import WordSwapCounterfit
    from datasets import YelpSentiment
    
    # @TODO: Running attack.py should parse args and run script-based attacks 
    #       (as opposed to code-based attacks)
    model = BertForSentimentClassification()
    
    attack = attacks.GreedyWordSwap(
        model,
        WordSwapCounterfit()
    )
    
    # attack.add_constraints(
        # constraints.syntax.LanguageTool(1),
        # constraints.semantics.UniversalSentenceEncoder(0.9, metric='cosine')
    # )
    
    yelp_data = YelpSentiment(n=2)
    # yelp_data = [
    #     (1, 'I hate this Restaurant!'), 
    #     (0, "Texas Jack's has amazing food.")
    # ]
    
    # attack.enable_visdom()
    attack.add_output_file(open('outputs/test.txt', 'w'))
    import sys
    attack.add_output_file(sys.stdout)
    
    attack.attack(yelp_data, n=10, shuffle=False)