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
        self.original_result.tokenized_text.free_memory()
        self.perturbed_result.tokenized_text.free_memory()
    
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
        print('t1:', t1)
        print('t2:',t2)
        
        if color_method is None:
            return t1.clean_text(), t2.clean_text()
        
        color_1 = self.original_result.get_text_color_input()
        color_2 = self.perturbed_result.get_text_color_perturbed()
        
        words_1 = []
        words_2 = []
        i1 = 0
        i2 = 0
        print("t1.attack_attrs['deletion_indices']", t1.attack_attrs['deletion_indices'])
        print("t2.attack_attrs['deletion_indices']", t2.attack_attrs['deletion_indices'])
        print('t1 deleted words:', [t1.words[x] for x in t2.attack_attrs['deletion_indices']])
        while i1 < len(t1.words) and i2 < len(t2.words):
            # show deletions
            while i1 in t2.attack_attrs['deletion_indices']:
                words_1.append(utils.color_text(t1.words[i1], color_1, color_method)) # @TODO color
                i1 += 1
            # show insertions
            while i2 in t2.attack_attrs['insertion_indices']:
                # @TODO this isnt right
                words_2.append(t2.words[i2])
                i2 += 1
            # show swaps
            word_1 = t1.words[i1]
            word_2 = t2.words[i2]
            print(i1,'/', i2,word_1,'/',word_2)
            if word_1 != word_2:
                words_1.append(utils.color_text(word_1, color_1, color_method))
                words_2.append(utils.color_text(word_2, color_2, color_method))
            else:
                words_1.append(word_1)
                words_2.append(word_2)
            i1 += 1
            i2 += 1
                
        return ' '.join(words_1), ' '.join(words_2)
        
