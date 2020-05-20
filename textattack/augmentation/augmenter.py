import random
import tqdm

from textattack.constraints.pre_transformation import PreTransformationConstraint
from textattack.shared.tokenized_text import TokenizedText

class Augmenter:
    """ 
    A class for performing data augmentation using TextAttack.
    
    Returns all possible transformations for a given string.
    
    Args:
        transformation (textattack.Transformation): the transformation
            that suggests new texts from an input.
        constraints: (list(textattack.Constraint)): constraints
            that each transformation must meet
        words_to_swap: (int): Number of words to swap per augmented example
        transformations_per_example: (int): Maximum number of augmentations
            per input
    """
    def __init__(self, transformation, constraints=[], words_to_swap=1, 
        transformations_per_example=1):
        self.transformation = transformation
        self.words_to_swap = words_to_swap
        self.transformations_per_example = transformations_per_example
        
        self.constraints = []
        self.pre_transformation_constraints = []
        for constraint in constraints:
            if isinstance(constraint, PreTransformationConstraint):
                self.pre_transformation_constraints.append(constraint)
            else:
                self.constraints.append(constraint)
    
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
            indices_to_modify = set(range(len(tokenized_text.words)))
            next_tokenized_text = tokenized_text
            for __ in range(self.words_to_swap):
                transformations = []
                for constraint in self.pre_transformation_constraints:
                    indices_to_modify = list(set(indices_to_modify) & constraint(tokenized_text, self))
                while not len(transformations):
                    # Loop until we find a valid transformation.
                    if not len(indices_to_modify):
                        # This occurs when we couldn't find valid transformations
                        # – either the constraints were too strict, or the number
                        # of words to swap was too high, or we were just plain 
                        # unlucky. In any event, don't throw an error, and just 
                        # don't return anything.
                        break
                    replacement_index = random.choice(indices_to_modify)
                    indices_to_modify.remove(replacement_index)
                    transformations = self.transformation(next_tokenized_text, indices_to_modify=[replacement_index])
                    # Get rid of transformations we already have
                    transformations = [t for t in transformations if t not in all_transformations]
                    # Filter out transformations that don't match the constraints.
                    transformations = self._filter_transformations(tokenized_text, transformations)
                if len(transformations):
                    next_tokenized_text = random.choice(transformations)
            all_transformations.add(next_tokenized_text)
        return [t.clean_text() for t in all_transformations]
    
    def augment_many(self, text_list):
        """
        Returns all possible augmentations of a list of strings according to
        `self.transformation`.
    
        Args:
            text_list (list(string)): a list of strings for data augmentation
            
        Returns a list(string) of augmented texts.
        """
        return [self.augment(text) for text in text_list]
        
    def augment_text_and_ids(self, text_list, id_list, show_progress=True):
        """ Supplements a list of text with more text data. Returns the augmented
            text along with the corresponding IDs for each augmented example.
        """
        if len(text_list) != len(id_list):
            raise ValueError('List of text must be same length as list of IDs')
        if self.transformations_per_example == 0:
            return text_list, id_list
        all_text_list = []
        all_id_list = []
        if show_progress:
            text_list = tqdm.tqdm(text_list, desc='Augmenting data...')
        for text, ids in zip(text_list, id_list):
            all_text_list.append(text)
            all_id_list.append(ids)
            augmented_texts = self.augment(text)
            all_text_list.extend
            all_text_list.extend([text] + augmented_texts)
            all_id_list.extend([ids] * (1 + len(augmented_texts)))
        return all_text_list, all_id_list
        
class DummyTokenizer:
    """ 
    A dummy tokenizer class. Data augmentation applies a transformation
    without querying a model, which means that tokenization is unnecessary.
    In this case, we pass a dummy tokenizer to `TokenizedText`. 
    """
    def encode(self, _):
        return []
