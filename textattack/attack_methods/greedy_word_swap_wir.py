import torch
import numpy as np

from .attack import Attack
from textattack.attack_results import FailedAttackResult, SuccessfulAttackResult

class GreedyWordSwapWIR(Attack):
    """
    An attack that greedily chooses from a list of possible perturbations in 
    order of index, after ranking indices by importance.
    
    Reimplementation of paper:
        Is BERT Really Robust? A Strong Baseline for Natural Language Attack on 
        Text Classification and Entailment by Jin et. al, 2019
        
        https://github.com/jind11/TextFooler 
        
    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation: The type of transformation.
        max_depth (:obj:`int`, optional): The maximum number of words to change. Defaults to 32. 
    """
    WIR_TO_REPLACEMENT_STR = {
        'unk': '[UNK]',
        'delete': '[DELETE]',
    }

    def __init__(self, goal_function, transformation, constraints=[], wir_method='unk', max_depth=32):
        super().__init__(goal_function, transformation, constraints=constraints)
        self.max_depth = max_depth
        try: 
            self.replacement_str = self.WIR_TO_REPLACEMENT_STR[wir_method]
        except KeyError:
            raise KeyError(f'Word Importance Ranking method {wir_method} not recognized.') 
        
    def attack_one(self, tokenized_text, correct_output):
        original_tokenized_text = tokenized_text
        num_words_changed = 0
       
        # Sort words by order of importance
        original_result = self.goal_function.get_results([tokenized_text], correct_output)[0]
        cur_score = original_result.score
        len_text = len(tokenized_text.words)
        
        leave_one_texts = \
            [tokenized_text.replace_word_at_index(i,self.replacement_str) for i in range(len_text)]
        leave_one_scores = np.array([result.score for result in \
            self.goal_function.get_results(leave_one_texts, correct_output)])
        index_order = (-leave_one_scores).argsort()

        new_tokenized_text = None
        new_text_label = None
        i = 0
        while ((self.max_depth is None) or num_words_changed <= self.max_depth) and i < len(index_order):
            transformed_text_candidates = self.get_transformations(
                tokenized_text,
                original_tokenized_text,
                indices_to_replace=[index_order[i]])
            i += 1
            if len(transformed_text_candidates) == 0:
                continue
            num_words_changed += 1
            results = sorted(self.goal_function.get_results(transformed_text_candidates, correct_output), 
                    key=lambda x: -x.score)
            # Skip swaps which don't improve the score
            if results[0].score > cur_score:
                cur_score = results[0].score
            else:
                continue
            # If we succeeded, return the index with best similarity.
            if results[0].succeeded:
                best_result = results[0]
                # @TODO: Use vectorwise operations
                max_similarity = -float('inf')
                for result in results:
                    if not result.succeeded:
                        break
                    candidate = result.tokenized_text
                    try:
                        similarity_score = candidate.attack_attrs['similarity_score']
                    except KeyError:
                        # If the attack was run without any similarity metrics, 
                        # candidates won't have a similarity score. In this
                        # case, break and return the candidate that changed
                        # the original score the most.
                        break
                    if similarity_score > max_similarity:
                        max_similarity = similarity_score
                        best_result = result
                return SuccessfulAttackResult( 
                    original_result,
                    best_result
                )
            else:
                tokenized_text = results[0].tokenized_text
        
        if len(results):
            return FailedAttackResult(original_result, results[0])
        else:
            return FailedAttackResult(original_result)
