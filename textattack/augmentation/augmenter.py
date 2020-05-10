import random

from textattack.shared.tokenized_text import TokenizedText

class Augmenter:
    """ A class for performing data augmentation using TextAttack.
    
        Returns all possible transformations for a given string.
        
        Args:
            transformation (textattack.Transformation): the transformation
                that suggests new texts from an input.
            constraints: (list(textattack.Constraint)): constraints
                that each transformation must meet
            words_to_swap (int): number of words to swap in each augmented 
                example
            transformations_per_example (int): number of transformations
                to return for each training example
    """
    def __init__(self, transformation, constraints=[], words_to_swap=1, 
        transformations_per_example=1):
        self.transformation = transformation
        self.constraints = constraints
        self.words_to_swap = words_to_swap
        self.transformations_per_example = transformations_per_example
    
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
        all_transformations = set()
        for _ in range(self.transformations_per_example):
            indices_to_replace = list(range(len(tokenized_text.words)))
            next_tokenized_text = tokenized_text
            for __ in range(self.words_to_swap):
                transformations = []
                while not len(transformations):
                    if not len(indices_to_replace):
                        # This occurs when we couldn't find valid transformations
                        # – either the constraints were too strict, or the number
                        # of words to swap was too high, or we were just plain 
                        # unlucky. In any event, don't throw an error, and just 
                        # don't return anything.
                        break
                    replacement_index = random.choice(indices_to_replace)
                    indices_to_replace.remove(replacement_index)
                    transformations = self.transformation(next_tokenized_text, indices_to_replace=[replacement_index])
                    # Get rid of transformations we already have
                    transformations = [t for t in transformations if t not in all_transformations]
                    # Filter out transformations that don't match the constraints.
                    transformations = self._filter_transformations(tokenized_text, transformations)
                if len(transformations):
                    next_tokenized_text = random.choice(transformations)
            all_transformations.add(next_tokenized_text)
        return [t.clean_text() for t in all_transformations]
    
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