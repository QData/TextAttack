from textattack import utils as utils

class AttackResult:
    """
    Result of an Attack run on a single (label, text_input) pair. 

    Args:
        original_text (str): The original text
        perturbed_text (str): The perturbed text resulting from the attack
        original_label (int): he classification label of the original text
        perturbed_label (int): The classification label of the perturbed text

    """
    def __init__(self, original_text, perturbed_text, original_label,
        perturbed_label):
        if original_text is None:
            raise ValueError('Attack original text cannot be None')
        if perturbed_text is None:
            raise ValueError('Attack perturbed text cannot be None')
        if original_label is None:
            raise ValueError('Attack original label cannot be None')
        if perturbed_label is None:
            raise ValueError('Attack perturbed label cannot be None')
        self.original_text = original_text
        self.perturbed_text = perturbed_text
        self.original_label = original_label
        self.perturbed_label = perturbed_label
        self.num_queries = 0

    def __data__(self, color_method=None):
        data = [self.result_str(color_method=color_method), 
                self.original_text.text,
                self.perturbed_text.text]
        if color_method is not None:
            data[1], data[2] = self.diff_color(color_method)
        return data
    
    def __str__(self, color_method=None):
        return '\n'.join(self.__data__(color_method=color_method))
   
    def result_str(self, color_method=None):
        orig_colored = utils.color_label(self.original_label, method=color_method)
        pert_colored = utils.color_label(self.perturbed_label, method=color_method)
        return orig_colored + '-->' + pert_colored

    def diff_color(self, color_method=None):
        """ 
        Highlights the difference between two texts using color.
        
        """
        t1 = self.original_text
        t2 = self.perturbed_text
        
        if color_method is None:
            return t1.text, t2.text

        words1 = t1.tokens
        words2 = t2.tokens
        
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
                new_w1s.append(utils.color(w1, c1, color_method))
                new_w2s.append(utils.color(w2, c2, color_method))
        
        t1 = self.original_text.replace_tokens_at_indices(new_is, new_w1s)
        t2 = self.original_text.replace_tokens_at_indices(new_is, new_w2s)
                
        return t1.text, t2.text
