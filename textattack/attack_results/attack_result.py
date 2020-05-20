from textattack.goal_function_results import GoalFunctionResult
from textattack.shared import utils

class AttackResult:
    """
    Result of an Attack run on a single (output, text_input) pair. 

    Args:
        original_result (GoalFunctionResult): Result of the goal function
            applied to the original text
        perturbed_result (GoalFunctionResult): Result of the goal function applied to the
            perturbed text. May or may not have been successful.
    """
    def __init__(self, original_result, perturbed_result):
        if original_result is None:
            raise ValueError('Attack original result cannot be None')
        elif not isinstance(original_result, GoalFunctionResult):
            raise TypeError(f'Invalid original goal function result: {original_text}')
        if perturbed_result is None:
            raise ValueError('Attack perturbed result cannot be None')
        elif not isinstance(perturbed_result, GoalFunctionResult):
            raise TypeError(f'Invalid perturbed goal function result: {perturbed_result}')
            
        self.original_result = original_result
        self.perturbed_result = perturbed_result
        self.num_queries = 0
        
        # We don't want the TokenizedText `ids` sticking around clogging up 
        # space on our devices. Delete them here, if they're still present,
        # because we won't need them anymore anyway.
        self.original_result.tokenized_text.delete_tensors()
        self.perturbed_result.tokenized_text.delete_tensors()
    
    def original_text(self):
        """ Returns the text portion of `self.original_result`. Helper method.
        """
        return self.original_result.tokenized_text.clean_text()
    
    def perturbed_text(self):
        """ Returns the text portion of `self.perturbed_result`. Helper method.
        """
        return self.original_result.tokenized_text.clean_text()

    def str_lines(self, color_method=None):
        """ A list of the lines to be printed for this result's string
            representation. """
        lines = [self.goal_function_result_str(color_method=color_method)]
        lines.extend(self.diff_color(color_method))
        return lines
    
    def __str__(self, color_method=None):
        return '\n'.join(self.str_lines(color_method=color_method))
   
    def goal_function_result_str(self, color_method=None):
        """
        Returns a string illustrating the results of the goal function.
        """
        orig_colored = self.original_result.get_colored_output(color_method) # @TODO add this method to goal function results
                                                                        # @TODO also display confidence
        pert_colored = self.perturbed_result.get_colored_output(color_method)
        return orig_colored + '-->' + pert_colored

    def diff_color(self, color_method=None):
        """ 
        Highlights the difference between two texts using color.
        
        """
        t1 = self.original_result.tokenized_text
        t2 = self.perturbed_result.tokenized_text
        
        if color_method is None:
            return t1.clean_text(), t2.clean_text()
        
        color_1 = self.original_result.get_text_color_input()
        color_2 = self.perturbed_result.get_text_color_perturbed()
        replaced_word_indices = []
        new_words_1 = []
        new_words_2 = []
        for i in range(min(len(t1.words), len(t2.words))):
            word_1 = t1.words[i]
            word_2 = t2.words[i]
            if word_1 != word_2:
                replaced_word_indices.append(i)
                new_words_1.append(utils.color_text(word_1, color_1, color_method))
                new_words_2.append(utils.color_text(word_2, color_2, color_method))
        
        t1 = self.original_result.tokenized_text.replace_words_at_indices(replaced_word_indices, 
            new_words_1)
        t2 = self.perturbed_result.tokenized_text.replace_words_at_indices(replaced_word_indices, 
            new_words_2)
                
        return t1.clean_text(), t2.clean_text()
        
