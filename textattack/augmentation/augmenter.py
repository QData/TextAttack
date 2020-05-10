from textattack.shared.tokenized_text import TokenizedText

class Augmenter:
    """ A class for performing data augmentation using TextAttack.
    
        Returns all possible transformations for a given string.
        
        Args:
            transformation (textattack.Transformation): the transformation
                that suggests new texts from an input.
            constraints: (list(textattack.Constraint)): constraints
                that each transformation must meet
    """
    def __init__(self, transformation, constraints=[]):
        self.transformation = transformation
        self.constraints = constraints
    
    def _filter_transformations(self, tokenized_text, transformations):
        """ Filters a list of `TokenizedText` objects to include only the ones 
            that pass `self.constraints`.
        """
        for C in self.constraints:
            if len(transformations) == 0: break
            transformations = C.call_many(tokenized_text, transformations, original_text=tokenized_text)
        return transformations
    
    def augment(self, text):
        """ Returns all possible augmentations of `text` according to 
            `self.transformation`.
        """
        tokenized_text = TokenizedText(text, DummyTokenizer())
        # Get potential transformations for text.
        transformations = self.transformation(tokenized_text)
        # Filter out transformations that don't match the constraints.
        transformations = self._filter_transformations(tokenized_text, transformations)
        return [t.clean_text() for t in transformations]
    
    def augment_many(self, text_list):
        """ Returns all possible augmentations of a list of strings according to
            `self.transformation`.
        """
        return [self.augment(text) for text in text_list]
        
class DummyTokenizer:
    """ A dummy tokenizer class. Data augmentation applies a transformation
        without querying a model, which means that tokenization is unnecessary.
        In this case, we pass a dummy tokenizer to `TokenizedText`. 
    """
    def encode(self, _):
        return []