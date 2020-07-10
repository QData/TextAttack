from abc import ABC, abstractmethod

from textattack.constraints import Constraint


class LanguageModelConstraint(Constraint, ABC):
    """Determines if two sentences have a swapped word that has a similar
    probability according to a language model.

    Args:
        max_log_prob_diff (float): the maximum decrease in log-probability
            in swapped words from `x` to `x_adv`
        compare_against_original (bool):  If `True`, compare new `x_adv` against the original `x`.
            Otherwise, compare it against the previous `x_adv`.
    """

    def __init__(self, max_log_prob_diff=None, compare_against_original=True):
        if max_log_prob_diff is None:
            raise ValueError("Must set max_log_prob_diff")
        self.max_log_prob_diff = max_log_prob_diff
        super().__init__(compare_against_original)

    @abstractmethod
    def get_log_probs_at_index(self, text_list, word_index):
        """Gets the log-probability of items in `text_list` at index
        `word_index` according to a language model."""
        raise NotImplementedError()

    def _check_constraint(self, transformed_text, reference_text):
        try:
            indices = transformed_text.attack_attrs["newly_modified_indices"]
        except KeyError:
            raise KeyError(
                "Cannot apply language model constraint without `newly_modified_indices`"
            )

        for i in indices:
            probs = self.get_log_probs_at_index((reference_text, transformed_text), i)
            if len(probs) != 2:
                raise ValueError(
                    f"Error: get_log_probs_at_index returned {len(probs)} values for 2 inputs"
                )
            ref_prob, transformed_prob = probs
            if transformed_prob <= ref_prob - self.max_log_prob_diff:
                return False

        return True

    def extra_repr_keys(self):
        return ["max_log_prob_diff"] + super().extra_repr_keys()
