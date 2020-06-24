"""
When WIR method is set to ``unk``, this is a
reimplementation of the search method from thepaper: 
Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and 
Entailment by Jin et. al, 2019. 
See https://arxiv.org/abs/1907.11932 and https://github.com/jind11/TextFooler.
"""

import numpy as np

from textattack.search_methods import SearchMethod
from textattack.shared.validators import transformation_consists_of_word_swaps


class GreedyWordSwapWIR(SearchMethod):
    """
    An attack that greedily chooses from a list of possible perturbations in 
    order of index, after ranking indices by importance.
        
    Args:
        wir_method: method for ranking most important words
        ascending: if True, ranks words from least-to-most important. (Default
            ranking shows the most important word first.)
    """

    def __init__(self, wir_method="unk", ascending=False):
        self.wir_method = wir_method
        self.ascending = ascending

    def _get_index_order(self, initial_result, texts):
        """ Queries model for list of attacked text objects ``text`` and
            ranks in order of descending score.
        """
        leave_one_results, search_over = self.get_goal_results(
            texts, initial_result.output
        )
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
            if self.ascending:
                index_order = (leave_one_scores).argsort()
            else:
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
            results, search_over = self.get_goal_results(
                transformed_text_candidates, initial_result.output
            )
            results = sorted(results, key=lambda x: -x.score)
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score:
                cur_result = results[0]
            else:
                continue
            # If we succeeded, return the index with best similarity.
            if cur_result.succeeded:
                best_result = cur_result
                # @TODO: Use vectorwise operations
                max_similarity = -float("inf")
                for result in results:
                    if not result.succeeded:
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
            Since it ranks words by their importance, GreedyWordSwapWIR is limited to word swaps transformations.
        """
        return transformation_consists_of_word_swaps(transformation)

    def extra_repr_keys(self):
        return ["wir_method"]
