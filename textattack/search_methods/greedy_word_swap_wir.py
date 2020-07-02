"""
When WIR method is set to ``unk``, this is a
reimplementation of the search method from the paper: 
Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and 
Entailment by Jin et. al, 2019. 
See https://arxiv.org/abs/1907.11932 and https://github.com/jind11/TextFooler.
"""

import numpy as np
import torch
from torch.nn.functional import softmax

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import (
    transformation_consists_of_word_swaps_and_deletions,
)


class GreedyWordSwapWIR(SearchMethod):
    """
    An attack that greedily chooses from a list of possible perturbations in 
    order of index, after ranking indices by importance.
        
    Args:
        wir_method: method for ranking most important words
    """

    def __init__(self, wir_method="unk"):
        self.wir_method = wir_method

    def _get_index_order(self, initial_result, texts):
        """ Queries model for list of attacked text objects ``text`` and
            ranks in order of descending score.
        """
        leave_one_results, search_over = self.get_goal_results(texts)
        leave_one_scores = np.array([result.score for result in leave_one_results])
        return leave_one_scores, search_over

    def _perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text
        cur_result = initial_result

        # Sort words by order of importance
        len_text = len(attacked_text.words)

        if self.wir_method == "unk":
            leave_one_texts = [
                attacked_text.replace_word_at_index(i, "[UNK]") for i in range(len_text)
            ]
            leave_one_scores, search_over = self._get_index_order(
                initial_result, leave_one_texts
            )
        elif self.wir_method == "pwws":
            # first, compute word saliency
            leave_one_texts = [
                attacked_text.replace_word_at_index(i, "[UNK]") for i in range(len_text)
            ]
            saliency_scores, search_over = self._get_index_order(
                initial_result, leave_one_texts
            )

            softmax_saliency_scores = softmax(torch.Tensor(saliency_scores)).numpy()

            # compute the largest change in score we can find by swapping each word
            delta_ps = []
            for idx in range(len_text):
                transformed_text_candidates = self.get_transformations(
                    cur_result.attacked_text,
                    original_text=initial_result.attacked_text,
                    indices_to_modify=[idx],
                )
                if not transformed_text_candidates:
                    # no valid synonym substitutions for this word
                    delta_ps.append(0.0)
                    continue
                swap_results, _ = self.get_goal_results(
                    transformed_text_candidates, initial_result.output
                )
                score_change = [result.score for result in swap_results]
                max_score_change = np.max(score_change)
                delta_ps.append(max_score_change)

            leave_one_scores = softmax_saliency_scores * np.array(delta_ps)

        elif self.wir_method == "delete":
            leave_one_texts = [
                attacked_text.delete_word_at_index(i) for i in range(len_text)
            ]
            leave_one_scores, search_over = self._get_index_order(
                initial_result, leave_one_texts
            )
        elif self.wir_method == "random":
            index_order = np.arange(len_text)
            np.random.shuffle(index_order)
            search_over = False

        if self.wir_method != "random":
            index_order = (-leave_one_scores).argsort()

        i = 0
        results = None
        while i < len(index_order) and not search_over:
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            )
            i += 1
            if len(transformed_text_candidates) == 0:
                continue
            results, search_over = self.get_goal_results(transformed_text_candidates)
            results = sorted(results, key=lambda x: -x.score)
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score:
                cur_result = results[0]
            else:
                continue
            # If we succeeded, return the index with best similarity.
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                best_result = cur_result
                # @TODO: Use vectorwise operations
                max_similarity = -float("inf")
                for result in results:
                    if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                        break
                    candidate = result.attacked_text
                    try:
                        similarity_score = candidate.attack_attrs["similarity_score"]
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

        return cur_result

    def check_transformation_compatibility(self, transformation):
        """
            Since it ranks words by their importance, GreedyWordSwapWIR is limited to word swap and deletion transformations.
        """
        return transformation_consists_of_word_swaps_and_deletions(transformation)

    def extra_repr_keys(self):
        return ["wir_method"]
