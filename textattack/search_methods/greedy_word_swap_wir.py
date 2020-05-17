import numpy as np

from textattack.search_methods import SearchMethod
from textattack.attack_results import FailedAttackResult, SuccessfulAttackResult

class GreedyWordSwapWIR(SearchMethod):
    """
    An attack that greedily chooses from a list of possible perturbations in 
    order of index, after ranking indices by importance.
    
    Reimplementation of paper:
        Is BERT Really Robust? A Strong Baseline for Natural Language Attack on 
        Text Classification and Entailment by Jin et. al, 2019
        
        https://github.com/jind11/TextFooler 
        
    Args:
        wir_method: method for ranking most important words
    """

    WIR_TO_REPLACEMENT_STR = {
        'unk': '[UNK]',
        'delete': '[DELETE]',
    }

    def __init__(self, wir_method='unk'):
        try: 
            self.replacement_str = self.WIR_TO_REPLACEMENT_STR[wir_method]
        except KeyError:
            raise KeyError(f'Word Importance Ranking method {wir_method} not recognized.') 
        
    def __call__(self, initial_result):
        original_tokenized_text = intial_result.tokenized_text
        cur_result = intial_result

        # Sort words by order of importance
        len_text = len(tokenized_text.words)
        
        leave_one_texts = \
            [tokenized_text.replace_word_at_index(i,self.replacement_str) for i in range(len_text)]
        leave_one_scores = np.array([result.score for result in \
            self.get_goal_results(leave_one_texts, intial_result.output)])
        index_order = (-leave_one_scores).argsort()

        i = 0
        while i < len(index_order):
            transformed_text_candidates = self.get_transformations(
                cur_result.tokenized_text,
                original_text=original_tokenized_text,
                indices_to_modify=[index_order[i]])
            i += 1
            if len(transformed_text_candidates) == 0:
                continue
            results = sorted(self.get_goal_results(transformed_text_candidates, intial_result.output), 
                    key=lambda x: -x.score)
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score:
                cur_result = results[0]
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
                return best_result
            else:
                cur_result = results[0]
       
        if results and len(results):
            return results[0]
        return initial_result
