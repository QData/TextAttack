"""
LanguageTool Grammar Checker
------------------------------
"""
import language_tool_python

from textattack.constraints import Constraint


class LanguageTool(Constraint):
    """Uses languagetool to determine if two sentences have the same number of
    grammatical erors. (https://languagetool.org/)

    Args:
        grammar_error_threshold (int): the number of additional errors permitted in `x_adv`
            relative to `x`
        compare_against_original (bool): If `True`, compare against the original text.
            Otherwise, compare against the most recent text.
    """

    def __init__(self, grammar_error_threshold=0, compare_against_original=True):
        super().__init__(compare_against_original)
        self.lang_tool = language_tool_python.LanguageTool("en-US")
        self.grammar_error_threshold = grammar_error_threshold
        self.grammar_error_cache = {}

    def get_errors(self, attacked_text, use_cache=False):
        text = attacked_text.text
        if use_cache:
            if text not in self.grammar_error_cache:
                self.grammar_error_cache[text] = len(self.lang_tool.check(text))
            return self.grammar_error_cache[text]
        else:
            return len(self.lang_tool.check(text))

    def _check_constraint(self, transformed_text, reference_text):
        original_num_errors = self.get_errors(reference_text, use_cache=True)
        errors_added = self.get_errors(transformed_text) - original_num_errors
        return errors_added <= self.grammar_error_threshold

    def extra_repr_keys(self):
        return ["grammar_error_threshold"] + super().extra_repr_keys()
