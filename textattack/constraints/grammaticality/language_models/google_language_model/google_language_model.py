from collections import defaultdict
import time

import numpy as np

from textattack.constraints import Constraint
from textattack.transformations import WordSwap

from .alzantot_goog_lm import GoogLMHelper


class GoogleLanguageModel(Constraint):
    """
    Constraint that uses the Google 1 Billion Words Language Model to 
    determine the difference in perplexity between x and x_adv. 

    Args:
        top_n (int):
        top_n_per_index (int):
        print_step (:obj:`bool`, optional): Whether to print each step. Defaults to False. 
        
    Returns:
        The :obj:`top_n` sentences.

    Raises:
        ValueError: If :obj:`top_n` or :obj:`top_n_per_index` are not provided. 
    """

    def __init__(self, top_n=None, top_n_per_index=None, print_step=False):
        if not (top_n or top_n_per_index):
            raise ValueError(
                "Cannot instantiate GoogleLanguageModel without top_n or top_n_per_index"
            )
        self.lm = GoogLMHelper()
        self.top_n = top_n
        self.top_n_per_index = top_n_per_index
        self.print_step = print_step

    def check_compatibility(self, transformation):
        return isinstance(transformation, WordSwap)

    def _check_constraint_many(
        self, transformed_texts, current_text, original_text=None
    ):
        """
        Returns the `top_n` of transformed_texts, as evaluated by the language 
        model. 
        """
        if not len(transformed_texts):
            return []

        def get_probs(current_text, transformed_texts):
            word_swap_index = current_text.first_word_diff_index(transformed_texts[0])
            prefix = current_text.words[word_swap_index - 1]
            swapped_words = np.array(
                [t.words[word_swap_index] for t in transformed_texts]
            )
            if self.print_step:
                print(prefix, swapped_words, suffix)
            probs = self.lm.get_words_probs(prefix, swapped_words)
            return probs

        # This creates a dictionary where each new key is initialized to [].
        word_swap_index_map = defaultdict(list)

        for idx, transformed_text in enumerate(transformed_texts):
            word_swap_index = current_text.first_word_diff_index(transformed_text)
            word_swap_index_map[word_swap_index].append((idx, transformed_text))

        probs = []
        for word_swap_index, item_list in word_swap_index_map.items():
            # zip(*some_list) is the inverse operator of zip!
            item_indices, this_transformed_texts = zip(*item_list)
            t1 = time.time()
            probs_of_swaps_at_index = list(
                zip(item_indices, get_probs(current_text, this_transformed_texts))
            )
            # Sort by probability in descending order and take the top n for this index.
            probs_of_swaps_at_index.sort(key=lambda x: -x[1])
            if self.top_n_per_index:
                probs_of_swaps_at_index = probs_of_swaps_at_index[
                    : self.top_n_per_index
                ]
            probs.extend(probs_of_swaps_at_index)
            t2 = time.time()
            if self.print_step:
                print(f"LM {len(item_list)} items in {t2-t1}s")

        # Probs is a list of (index, prob) where index is the corresponding
        # position in transformed_texts.
        probs.sort(key=lambda x: x[0])

        # Now that they're in order, reduce to just a list of probabilities.
        probs = np.array(list(map(lambda x: x[1], probs)))

        # Get the indices of the maximum elements.
        max_el_indices = np.argsort(-probs)
        if self.top_n:
            max_el_indices = max_el_indices[: self.top_n]

        # Put indices in order, now, so that the examples are returned in the
        # same order they were passed in.
        max_el_indices.sort()

        return [transformed_texts[i] for i in max_el_indices]

    def __call__(self, x, x_adv):
        raise NotImplementedError()

    def extra_repr_keys(self):
        return ["top_n", "top_n_per_index"]
