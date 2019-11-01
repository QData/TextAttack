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
    
    def get_errors(self, tokenized_text, use_cache=False):
        text = tokenized_text.text
        if use_cache:
            if text not in self.grammar_error_cache:
                self.grammar_error_cache[text] = len(self.lang_tool.check(text))
            return self.grammar_error_cache[text]
        else:
            return len(self.lang_tool.check(text))
    
    def call_many(self, x, x_adv_list, original_text=None):
        return [x_adv for x_adv in x_adv_list 
                if self.__call__(x, x_adv, original_text=original_text)]
    
    def __call__(self, x, x_adv, original_text=None):
        original_num_errors = self.get_errors(original_text, use_cache=True)
        errors_added = self.get_errors(x_adv) - original_num_errors
        return errors_added <= self.threshold