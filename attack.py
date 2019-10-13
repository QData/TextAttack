# @TODO all files should have some copyright header or something

"""" @TODOs:
    - Example of using with Pytorch and Tensorflow model.
    - sphinx documentation
    - add recipes with state-of-the-art attacks
    - add unit tests
    - add pep8 standard
    - upload sample models and datasets
"""

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
        print('results:', len(results))
        
        for output_file in self.output_files:
            for result in results:
                output_file.write(result.original_text.text + '\n')
                output_file.write(str(result.original_label) + '\n')
                output_file.write(result.perturbed_text.text + '\n')
                output_file.write(str(result.perturbed_label) + '\n')
                output_file.write('\n')
        
        return results

class AttackResult:
    def __init__(self, original_text, perturbed_text, original_label,
        perturbed_label):
        self.original_text = original_text
        self.perturbed_text = perturbed_text
        self.original_label = original_label
        self.perturbed_label = perturbed_label

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
    
    yelp_data = YelpSentiment(N=32)
    # yelp_data = [
    #     (1, 'I hate this Restaurant!'), 
    #     (0, "Texas Jack's has amazing food.")
    # ]
    
    # attack.enable_visdom()
    attack.add_output_file(open('outputs/test.txt', 'w'))
    import sys
    attack.add_output_file(sys.stdout)
    
    attack.attack(yelp_data, n=10, shuffle=False)