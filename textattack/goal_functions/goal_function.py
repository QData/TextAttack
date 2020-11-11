"""
goal_function: Goal functions determine if an attack has been successful.
==============================================================================
"""

from abc import ABC, abstractmethod

import lru

from textattack.goal_function_results.goal_function_result import (
    GoalFunctionResultStatus,
)
from textattack.shared import validators
from textattack.shared.utils import default_class_repr


class GoalFunction(ABC):
    """Evaluates how well a perturbed attacked_text object is achieving a
    specified goal.

    Args:
        model_wrapper: The model used for evaluation.
        maximizable: Whether the goal function is maximizable, as opposed to a boolean result
            of success or failure.
        query_budget (float): The maximum number of model queries allowed.
        model_cache_size (int): The maximum number of items to keep in the model
            results cache at once
    """

    def __init__(
        self,
        model_wrapper,
        maximizable=False,
        use_cache=True,
        query_budget=float("inf"),
        model_cache_size=2 ** 20,
    ):
        validators.validate_model_goal_function_compatibility(
            self.__class__, model_wrapper.model.__class__
        )
        self.model = model_wrapper
        self.maximizable = maximizable
        self.use_cache = use_cache
        self.query_budget = query_budget
        if self.use_cache:
            self._call_model_cache = lru.LRU(model_cache_size)
        else:
            self._call_model_cache = None

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()

    def init_attack_example(self, attacked_text, ground_truth_output):
        """Called before attacking ``attacked_text`` to 'reset' the goal
        function and set properties for this example."""
        self.initial_attacked_text = attacked_text
        self.ground_truth_output = ground_truth_output
        self.num_queries = 0
        result, _ = self.get_result(attacked_text, check_skip=True)
        return result, _

    def get_output(self, attacked_text):
        """Returns output for display based on the result of calling the
        model."""
        return self._get_displayed_output(self._call_model([attacked_text])[0])

    def get_result(self, attacked_text, **kwargs):
        """A helper method that queries ``self.get_results`` with a single
        ``AttackedText`` object."""
        results, search_over = self.get_results([attacked_text], **kwargs)
        result = results[0] if len(results) else None
        return result, search_over

    def get_results(self, attacked_text_list, check_skip=False):
        """For each attacked_text object in attacked_text_list, returns a
        result consisting of whether or not the goal has been achieved, the
        output for display purposes, and a score.

        Additionally returns whether the search is over due to the query
        budget.
        """
        results = []
        if self.query_budget < float("inf"):
            queries_left = self.query_budget - self.num_queries
            attacked_text_list = attacked_text_list[:queries_left]
        self.num_queries += len(attacked_text_list)
        model_outputs = self._call_model(attacked_text_list)
        for attacked_text, raw_output in zip(attacked_text_list, model_outputs):
            displayed_output = self._get_displayed_output(raw_output)
            goal_status = self._get_goal_status(
                raw_output, attacked_text, check_skip=check_skip
            )
            goal_function_score = self._get_score(raw_output, attacked_text)
            results.append(
                self._goal_function_result_type()(
                    attacked_text,
                    raw_output,
                    displayed_output,
                    goal_status,
                    goal_function_score,
                    self.num_queries,
                    self.ground_truth_output,
                )
            )
        return results, self.num_queries == self.query_budget

    def _get_goal_status(self, model_output, attacked_text, check_skip=False):
        should_skip = check_skip and self._should_skip(model_output, attacked_text)
        if should_skip:
            return GoalFunctionResultStatus.SKIPPED
        if self.maximizable:
            return GoalFunctionResultStatus.MAXIMIZING
        if self._is_goal_complete(model_output, attacked_text):
            return GoalFunctionResultStatus.SUCCEEDED
        return GoalFunctionResultStatus.SEARCHING

    @abstractmethod
    def _is_goal_complete(self, model_output, attacked_text):
        raise NotImplementedError()

    def _should_skip(self, model_output, attacked_text):
        return self._is_goal_complete(model_output, attacked_text)

    @abstractmethod
    def _get_score(self, model_output, attacked_text):
        raise NotImplementedError()

    def _get_displayed_output(self, raw_output):
        return raw_output

    @abstractmethod
    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        raise NotImplementedError()

    @abstractmethod
    def _process_model_outputs(self, inputs, outputs):
        """Processes and validates a list of model outputs.

        This is a task-dependent operation. For example, classification
        outputs need to make sure they have a softmax applied.
        """
        raise NotImplementedError()

    def _call_model_uncached(self, attacked_text_list):
        """Queries model and returns outputs for a list of AttackedText
        objects."""
        if not len(attacked_text_list):
            return []

        inputs = [at.tokenizer_input for at in attacked_text_list]

        outputs = self.model(inputs)

        assert len(inputs) == len(
            outputs
        ), f"Got {len(outputs)} outputs for {len(inputs)} inputs"

        return self._process_model_outputs(attacked_text_list, outputs)

    def _call_model(self, attacked_text_list):
        """Gets predictions for a list of ``AttackedText`` objects.

        Gets prediction from cache if possible. If prediction is not in
        the cache, queries model and stores prediction in cache.
        """
        if not self.use_cache:
            return self._call_model_uncached(attacked_text_list)
        else:
            uncached_list = []
            for text in attacked_text_list:
                if text in self._call_model_cache:
                    # Re-write value in cache. This moves the key to the top of the
                    # LRU cache and prevents the unlikely event that the text
                    # is overwritten when we store the inputs from `uncached_list`.
                    self._call_model_cache[text] = self._call_model_cache[text]
                else:
                    uncached_list.append(text)
            uncached_list = [
                text
                for text in attacked_text_list
                if text not in self._call_model_cache
            ]
            outputs = self._call_model_uncached(uncached_list)
            for text, output in zip(uncached_list, outputs):
                self._call_model_cache[text] = output
            all_outputs = [self._call_model_cache[text] for text in attacked_text_list]
            return all_outputs

    def extra_repr_keys(self):
        attrs = []
        if self.query_budget < float("inf"):
            attrs.append("query_budget")
        if self.maximizable:
            attrs.append("maximizable")
        return attrs

    __repr__ = __str__ = default_class_repr
