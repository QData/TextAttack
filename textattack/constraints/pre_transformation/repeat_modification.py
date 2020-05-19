from textattack.shared.utils import default_class_repr
from textattack.constraints.pre_transformation import PreTransformationConstraint

class RepeatModification(PreTransformationConstraint):
    """ 
    A constraint disallowing the modification of words which have already been modified.
    """
   
    def _get_modifiable_indices(self, tokenized_text):
        """ Returns the word indices in x which are able to be deleted """
        try:
            return set(range(len(tokenized_text.words))) - tokenized_text.attack_attrs['modified_indices'] 
        except KeyError:
            raise KeyError('`modified_indices` in attack_attrs required for RepeatModification constraint.')

