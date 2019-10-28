from alzantot_goog_lm import GoogLMHelper

class GoogleLanguageModel(Constraint):
    """ Constraint that uses the Google 1 Billion Words Language Model to 
        determine the difference in perplexity between x and x_adv. 
    """
    def __init__(self):
        self.lm = GoogLMHelper()
    
    def call_many(self, x, x_adv_list):
    
    def __call__(self, x, x_adv):