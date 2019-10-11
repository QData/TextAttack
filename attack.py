# @TODO all files should have some copyright header or something

"""" @TODOs:
    - Example of using with Pytorch and Tensorflow model.
    - sphinx documentation
"""

class Attack:
    
    def __init__(model, tokenizer, search_method, transformation):
        """ Initialize an attack object.
        
        Attacks can be run multiple times
        
         @TODO should `tokenizer` be an additional parameter or should
            we assume every model has a .tokenizer ?
        """
        self.model = model
        self.tokenizer = tokenizer
        self.attack_method = AttackMethod(search_method, transformation)
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
    
    def output_to_file(self, file):
        """ When attack runs, it will output to this file. """
        self.output_files.append(file)
    
    def run(self, dataset, n=None, shuffle=False):
        """ Runs an attack on some data and outputs results.
        
            - dataset: an iterable of (label, text) pairs
        """
        if shuffle:
            random.shuffle(dataset)
        
        _i = 0
        results = []
        for label, text in dataset:
            result = self.attack_method
            _i += 1
            if _i > n:
                break
    
    

if __name__ == '__main__':
    from .models import BertForSentimentClassification
    from .search_methods import Greedy
    from .transformations import SynonymSwap
    from .datasets import YelpSentiment
    
    # @TODO this should parse args and run script-based attacks 
    #       (as opposed to code-based attacks)
    model = BertForSentimentClassification()
    tokenizer = model.tokenizer
    
    attack_search_method = SynonymSwap(Greedy())
    attack = Attack(
        model,
        tokenizer,
        attack_method
    )
    
    attack.add_constraints(
        # .constraints.syntax.LanguageTool(1),
        # .constraints.semantics.UniversalSentenceEncoder(0.9, metric='cosine')
    )
    
    yelp_data = YelpSentiment()
    
    # attack.enable_visdom()
    attack.output_to_file(open('outputs/test.txt', 'w'))
    
    attack.run(yelp_data, n=10, shuffle=False)