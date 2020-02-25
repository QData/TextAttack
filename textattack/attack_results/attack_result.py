from textattack.shared import utils

class AttackResult:
    """
    Result of an Attack run on a single (output, text_input) pair. 

    Args:
        original_text (str): The original text
        perturbed_text (str): The perturbed text resulting from the attack
        original_output (int): he classification output of the original text
        perturbed_output (int): The classification output of the perturbed text

    """
    def __init__(self, original_text, perturbed_text, original_output,
        perturbed_output, orig_score=None, perturbed_score=None):
        if original_text is None:
            raise ValueError('Attack original text cannot be None')
        if perturbed_text is None:
            raise ValueError('Attack perturbed text cannot be None')
        if original_output is None:
            raise ValueError('Attack original output cannot be None')
        if perturbed_output is None:
            raise ValueError('Attack perturbed output cannot be None')
        self.original_text = original_text
        self.perturbed_text = perturbed_text
        self.original_output = original_output
        self.perturbed_output = perturbed_output
        self.orig_score = orig_score
        self.perturbed_score = perturbed_score
        self.num_queries = 0
        
        # We don't want the TokenizedText `ids` sticking around clogging up 
        # space on our devices. Delete them here, if they're still present,
        # because we won't need them anymore anyway.
        self.original_text.delete_tensors()
        self.perturbed_text.delete_tensors()

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
        # @TODO add comment to distinguish this from __str__
        orig_colored = utils.color_label(self.original_output, method=color_method)
        pert_colored = utils.color_label(self.perturbed_output, method=color_method)
        return orig_colored + '-->' + pert_colored

    def diff_color(self, color_method=None):
        """ 
        Highlights the difference between two texts using color.
        
        """
        t1 = self.original_text
        t2 = self.perturbed_text
        
        if color_method is None:
            return t1.text, t2.text

        words1 = t1.words
        words2 = t2.words
        
        c1 = utils.color_from_label(self.original_output)
        c2 = utils.color_from_label(self.perturbed_output)
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
        
        t1 = self.original_text.replace_words_at_indices(new_is, new_w1s)
        t2 = self.original_text.replace_words_at_indices(new_is, new_w2s)
                
        return t1.clean_text(), t2.clean_text()
