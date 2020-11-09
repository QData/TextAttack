"""
Attack Result Class
====================
"""

from abc import ABC

from textattack.goal_function_results import GoalFunctionResult
from textattack.shared import utils


class AttackResult(ABC):
    """Result of an Attack run on a single (output, text_input) pair.

    Args:
        original_result (GoalFunctionResult): Result of the goal function
            applied to the original text
        perturbed_result (GoalFunctionResult): Result of the goal function applied to the
            perturbed text. May or may not have been successful.
    """

    def __init__(self, original_result, perturbed_result):
        if original_result is None:
            raise ValueError("Attack original result cannot be None")
        elif not isinstance(original_result, GoalFunctionResult):
            raise TypeError(f"Invalid original goal function result: {original_result}")
        if perturbed_result is None:
            raise ValueError("Attack perturbed result cannot be None")
        elif not isinstance(perturbed_result, GoalFunctionResult):
            raise TypeError(
                f"Invalid perturbed goal function result: {perturbed_result}"
            )

        self.original_result = original_result
        self.perturbed_result = perturbed_result
        self.num_queries = perturbed_result.num_queries

        # We don't want the AttackedText attributes sticking around clogging up
        # space on our devices. Delete them here, if they're still present,
        # because we won't need them anymore anyway.
        self.original_result.attacked_text.free_memory()
        self.perturbed_result.attacked_text.free_memory()

    def original_text(self, color_method=None):
        """Returns the text portion of `self.original_result`.

        Helper method.
        """
        return self.original_result.attacked_text.printable_text(
            key_color=("bold", "underline"), key_color_method=color_method
        )

    def perturbed_text(self, color_method=None):
        """Returns the text portion of `self.perturbed_result`.

        Helper method.
        """
        return self.perturbed_result.attacked_text.printable_text(
            key_color=("bold", "underline"), key_color_method=color_method
        )

    def str_lines(self, color_method=None):
        """A list of the lines to be printed for this result's string
        representation."""
        lines = [self.goal_function_result_str(color_method=color_method)]
        lines.extend(self.diff_color(color_method))
        return lines

    def __str__(self, color_method=None):
        return "\n\n".join(self.str_lines(color_method=color_method))

    def goal_function_result_str(self, color_method=None):
        """Returns a string illustrating the results of the goal function."""
        orig_colored = self.original_result.get_colored_output(color_method)
        pert_colored = self.perturbed_result.get_colored_output(color_method)
        return orig_colored + " --> " + pert_colored

    def diff_color(self, color_method=None):
        """Highlights the difference between two texts using color.

        Has to account for deletions and insertions from original text to
        perturbed. Relies on the index map stored in
        ``self.original_result.attacked_text.attack_attrs["original_index_map"]``.
        """
        t1 = self.original_result.attacked_text
        t2 = self.perturbed_result.attacked_text

        if color_method is None:
            return t1.printable_text(), t2.printable_text()

        color_1 = self.original_result.get_text_color_input()
        color_2 = self.perturbed_result.get_text_color_perturbed()

        # iterate through and count equal/unequal words
        words_1_idxs = []
        t2_equal_idxs = set()
        original_index_map = t2.attack_attrs["original_index_map"]
        for t1_idx, t2_idx in enumerate(original_index_map):
            if t2_idx == -1:
                # add words in t1 that are not in t2
                words_1_idxs.append(t1_idx)
            else:
                w1 = t1.words[t1_idx]
                w2 = t2.words[t2_idx]
                if w1 == w2:
                    t2_equal_idxs.add(t2_idx)
                else:
                    words_1_idxs.append(t1_idx)

        # words to color in t2 are all the words that didn't have an equal,
        # mapped word in t1
        words_2_idxs = list(sorted(set(range(t2.num_words)) - t2_equal_idxs))

        # make lists of colored words
        words_1 = [t1.words[i] for i in words_1_idxs]
        words_1 = [utils.color_text(w, color_1, color_method) for w in words_1]
        words_2 = [t2.words[i] for i in words_2_idxs]
        words_2 = [utils.color_text(w, color_2, color_method) for w in words_2]

        t1 = self.original_result.attacked_text.replace_words_at_indices(
            words_1_idxs, words_1
        )
        t2 = self.perturbed_result.attacked_text.replace_words_at_indices(
            words_2_idxs, words_2
        )

        key_color = ("bold", "underline")
        return (
            t1.printable_text(key_color=key_color, key_color_method=color_method),
            t2.printable_text(key_color=key_color, key_color_method=color_method),
        )
