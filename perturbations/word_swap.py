from .perturbation import Perturbation
from constraints import TextConstraint, WordConstraint

class WordSwap(Perturbation):
    """ An abstract class that takes a sentence and perturbs it by replacing
        some of its words.
        
        Other classes can achieve this by inheriting from WordSwap and 
        overriding self._get_replacement_words.
    """
    # def __init__(self):
        # print('WordSwap init')
        # super().__init__()
    
    def add_constraints(self, constraints):
        """ Add multiple constraints.
        """
        for constraint in constraints:
            if not (isinstance(constraint, TextConstraint) or isinstance(constraint, WordConstraint)):
                raise ValueError('Cannot add constraint of type', type(constraint))
            self.add_constraint(constraint)
            
    def _get_replacement_words(self, word):
        raise NotImplementedError()
    
    def _filter_perturbations(self, original_text, perturbations, word_swaps):
        """ Filters a list of perturbations based on internal constraints. """
        
        """ Filters a list of perturbations based on internal constraints. """
        good_perturbations = []
        for p in perturbations:
            meets_constraints = True
            for c in self.constraints:
                if isinstance(c, TextConstraint):
                    meets_constraints = meets_constraints and c(original_text, p)
                elif isinstance(c, WordConstraint):
                    meets_constraints = meets_constraints and c(ow, nw)
                if not meets_constraints: break
            if meets_constraints: good_perturbations.append(p)
        return good_perturbations
    
    def perturb(self, tokenized_text, indices_to_replace=None):
        """ Returns a list of all possible perturbations for `text`.
            
            If indices_to_replace is set, only replaces words at those
                indices.
        """
        words = tokenized_text.words()
        if not indices_to_replace:
            indices_to_replace = list(range(len(words)))
        
        perturbations = []
        word_swaps = []
        for i in indices_to_replace:
            word_to_replace = words[i]
            replacement_words = self._get_replacement_words(word_to_replace)
            new_tokenized_texts = []
            for r in replacement_words:
                new_tokenized_texts.append(tokenized_text.replace_word_at_index(i, r))
                word_swaps.append((word_to_replace, r))
            perturbations.extend(new_tokenized_texts)
        
        return self._filter_perturbations(tokenized_text, perturbations, word_swaps)