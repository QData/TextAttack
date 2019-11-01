import language_check

from textattack.constraints import Constraint

class LanguageTool(Constraint):
    """ 
        Uses languagetool to determine if two sentences have the same
        number of typos. 
        (https://languagetool.org/)
        
        Args:
            threshold (int): the number of additional errors permitted in x_adv
                relative to x
    """
    
    def __init__(self, threshold=0):
        self.lang_tool = language_check.LanguageTool("en-US")
        self.threshold = threshold
        self.grammar_error_cache = {}
    
    def get_errors(self, tokenized_text):
        text = tokenized_text.text
        if text not in self.grammar_error_cache:
            num_errors = len(self.lang_tool.check(text))
            self.grammar_error_cache[text] = num_errors
        return self.grammar_error_cache[text]
    
    def call_many(self, x, x_adv_list, original_text=None):
        return [x_adv for x_adv in x_adv_list if self.__call__(x, x_adv)]
    
    def __call__(self, x, x_adv):
        errors_added = self.get_errors(x_adv) - self.get_errors(x)
        print(self.get_errors(x_adv),'/',self.get_errors(x))
        print('\t',errors_added)
        return errors_added <= self.threshold