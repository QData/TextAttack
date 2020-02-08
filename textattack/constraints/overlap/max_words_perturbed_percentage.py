from textattack.constraints import Constraint

class MaximumWordsPerturbedPercentage(Constraint):
    """ 
    A constraint representing a maximum allowed perturbed words percentage.

    """
    
    def __init__(self, max_percent=20):
        self.max_percent = max_percent

    def call_many(self, x, x_adv_list, original_text=None, **kwargs):
        """
        Filters x_adv_list to x_adv where C(x,x_adv) is true.

        Args:
            x:
            x_adv_list:
            original_text(:obj:`type`, optional): Defaults to None. 

        """
        return [x_adv for x_adv in x_adv_list 
                if self.__call__(x, x_adv, original_text=original_text)]
    
    def __call__(self, x, x_adv, original_text=None):
        if not original_text:
            return True
        num_words_diff = len(x_adv.all_words_diff(original_text))
        min_length = min(len(x_adv), len(original_text))
        max_perturbed = int(min_length * (self.max_percent / 100) + 0.5)
        return num_words_diff <= max_perturbed

