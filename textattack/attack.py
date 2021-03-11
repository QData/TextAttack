"""
Attack: TextAttack builds attacks from four components:
========================================================

- `Goal Functions <../attacks/goal_function.html>`__ stipulate the goal of the attack, like to change the prediction score of a classification model, or to change all of the words in a translation output.
- `Constraints <../attacks/constraint.html>`__ determine if a potential perturbation is valid with respect to the original input.
- `Transformations <../attacks/transformation.html>`__ take a text input and transform it by inserting and deleting characters, words, and/or phrases.
- `Search Methods <../attacks/search_method.html>`__ explore the space of possible **transformations** within the defined **constraints** and attempt to find a successful perturbation which satisfies the **goal function**.

The ``Attack`` class represents an adversarial attack composed of a goal function, search method, transformation, and constraints.
"""

import lru
import torch

import textattack
from textattack.attack_results import (
    FailedAttackResult,
    MaximizedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)
from textattack.constraints import Constraint, PreTransformationConstraint
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.goal_functions import GoalFunction
from textattack.models.wrappers import ModelWrapper
from textattack.search_methods import SearchMethod
from textattack.shared import AttackedText, utils
from textattack.transformations import CompositeTransformation, Transformation


class Attack:
    """An attack generates adversarial examples on text.

    An attack is comprised of a search method, goal function,
    a transformation, and a set of one or more linguistic constraints that
    successful examples must meet.

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        constraints: A list of constraints to add to the attack, defining which perturbations are valid.
        transformation: The transformation applied at each step of the attack.
        search_method: A strategy for exploring the search space of possible perturbations
        transformation_cache_size (int): the number of items to keep in the transformations cache
        constraint_cache_size (int): the number of items to keep in the constraints cache
    """

    def __init__(
        self,
        goal_function=None,
        constraints=[],
        transformation=None,
        search_method=None,
        transformation_cache_size=2 ** 15,
        constraint_cache_size=2 ** 15,
    ):
        """Initialize an attack object.

        Attacks can be run multiple times.
        """
        self.goal_function = goal_function
        if not self.goal_function:
            raise NameError(
                "Cannot instantiate attack without self.goal_function for predictions"
            )
        self.search_method = search_method
        if not self.search_method:
            raise NameError("Cannot instantiate attack without search method")
        self.transformation = transformation
        if not self.transformation:
            raise NameError("Cannot instantiate attack without transformation")
        self.is_black_box = (
            getattr(transformation, "is_black_box", True) and search_method.is_black_box
        )

        if not self.search_method.check_transformation_compatibility(
            self.transformation
        ):
            raise ValueError(
                f"SearchMethod {self.search_method} incompatible with transformation {self.transformation}"
            )

        self.constraints = []
        self.pre_transformation_constraints = []
        for constraint in constraints:
            if isinstance(
                constraint,
                textattack.constraints.PreTransformationConstraint,
            ):
                self.pre_transformation_constraints.append(constraint)
            else:
                self.constraints.append(constraint)

        # Check if we can use transformation cache for our transformation.
        if not self.transformation.deterministic:
            self.use_transformation_cache = False
        elif isinstance(self.transformation, CompositeTransformation):
            self.use_transformation_cache = True
            for t in self.transformation.transformations:
                if not t.deterministic:
                    self.use_transformation_cache = False
                    break
        else:
            self.use_transformation_cache = True
        self.transformation_cache_size = transformation_cache_size
        self.transformation_cache = lru.LRU(transformation_cache_size)

        self.constraint_cache_size = constraint_cache_size
        self.constraints_cache = lru.LRU(constraint_cache_size)

        # Give search method access to functions for getting transformations and evaluating them
        self.search_method.get_transformations = self.get_transformations
        # Give search method access to self.goal_function for model query count, etc.
        self.search_method.goal_function = self.goal_function
        # The search method only needs access to the first argument. The second is only used
        # by the attack class when checking whether to skip the sample
        self.search_method.get_goal_results = self.goal_function.get_results

        self.search_method.filter_transformations = self.filter_transformations

    def clear_cache(self, recursive=True):
        self.constraints_cache.clear()
        if self.use_transformation_cache:
            self.transformation_cache.clear()
        if recursive:
            self.goal_function.clear_cache()
            for constraint in self.constraints:
                if hasattr(constraint, "clear_cache"):
                    constraint.clear_cache()

    def cpu(self):
        """Move any models that are part of Attack to CPU."""
        visited = set()

        def to_cpu(obj):
            visited.add(id(obj))
            if isinstance(obj, torch.nn.Module):
                obj.cpu()
            elif isinstance(
                obj,
                (
                    Attack,
                    GoalFunction,
                    Transformation,
                    SearchMethod,
                    Constraint,
                    PreTransformationConstraint,
                    ModelWrapper,
                ),
            ):
                for key in obj.__dict__:
                    s_obj = obj.__dict__[key]
                    if id(s_obj) not in visited:
                        to_cpu(s_obj)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    if id(item) not in visited and isinstance(
                        item, (Transformation, Constraint, PreTransformationConstraint)
                    ):
                        to_cpu(item)

        to_cpu(self)

    def cuda(self):
        """Move any models that are part of Attack to GPU."""
        visited = set()

        def to_cuda(obj):
            visited.add(id(obj))
            if isinstance(obj, torch.nn.Module):
                obj.to(textattack.shared.utils.device)
            elif isinstance(
                obj,
                (
                    Attack,
                    GoalFunction,
                    Transformation,
                    SearchMethod,
                    Constraint,
                    PreTransformationConstraint,
                    ModelWrapper,
                ),
            ):
                for key in obj.__dict__:
                    s_obj = obj.__dict__[key]
                    if id(s_obj) not in visited:
                        to_cuda(s_obj)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    if id(item) not in visited and isinstance(
                        item, (Transformation, Constraint, PreTransformationConstraint)
                    ):
                        to_cuda(item)

        to_cuda(self)

    def _get_transformations_uncached(self, current_text, original_text=None, **kwargs):
        """Applies ``self.transformation`` to ``text``, then filters the list
        of possible transformations through the applicable constraints.

        Args:
            current_text: The current ``AttackedText`` on which to perform the transformations.
            original_text: The original ``AttackedText`` from which the attack started.
        Returns:
            A filtered list of transformations where each transformation matches the constraints
        """
        transformed_texts = self.transformation(
            current_text,
            pre_transformation_constraints=self.pre_transformation_constraints,
            **kwargs,
        )

        return transformed_texts

    def get_transformations(self, current_text, original_text=None, **kwargs):
        """Applies ``self.transformation`` to ``text``, then filters the list
        of possible transformations through the applicable constraints.

        Args:
            current_text: The current ``AttackedText`` on which to perform the transformations.
            original_text: The original ``AttackedText`` from which the attack started.
        Returns:
            A filtered list of transformations where each transformation matches the constraints
        """
        if not self.transformation:
            raise RuntimeError(
                "Cannot call `get_transformations` without a transformation."
            )

        if self.use_transformation_cache:
            cache_key = tuple([current_text] + sorted(kwargs.items()))
            if utils.hashable(cache_key) and cache_key in self.transformation_cache:
                # promote transformed_text to the top of the LRU cache
                self.transformation_cache[cache_key] = self.transformation_cache[
                    cache_key
                ]
                transformed_texts = list(self.transformation_cache[cache_key])
            else:
                transformed_texts = self._get_transformations_uncached(
                    current_text, original_text, **kwargs
                )
                if utils.hashable(cache_key):
                    self.transformation_cache[cache_key] = tuple(transformed_texts)
        else:
            transformed_texts = self._get_transformations_uncached(
                current_text, original_text, **kwargs
            )

        return self.filter_transformations(
            transformed_texts, current_text, original_text
        )

    def _filter_transformations_uncached(
        self, transformed_texts, current_text, original_text=None
    ):
        """Filters a list of potential transformaed texts based on
        ``self.constraints``

        Args:
            transformed_texts: A list of candidate transformed ``AttackedText`` to filter.
            current_text: The current ``AttackedText`` on which the transformation was applied.
            original_text: The original ``AttackedText`` from which the attack started.
        """
        filtered_texts = transformed_texts[:]
        for C in self.constraints:
            if len(filtered_texts) == 0:
                break
            if C.compare_against_original:
                if not original_text:
                    raise ValueError(
                        f"Missing `original_text` argument when constraint {type(C)} is set to compare against `original_text`"
                    )

                filtered_texts = C.call_many(filtered_texts, original_text)
            else:
                filtered_texts = C.call_many(filtered_texts, current_text)
        # Default to false for all original transformations.
        for original_transformed_text in transformed_texts:
            self.constraints_cache[(current_text, original_transformed_text)] = False
        # Set unfiltered transformations to True in the cache.
        for filtered_text in filtered_texts:
            self.constraints_cache[(current_text, filtered_text)] = True
        return filtered_texts

    def filter_transformations(
        self, transformed_texts, current_text, original_text=None
    ):
        """Filters a list of potential transformed texts based on
        ``self.constraints`` Utilizes an LRU cache to attempt to avoid
        recomputing common transformations.

        Args:
            transformed_texts: A list of candidate transformed ``AttackedText`` to filter.
            current_text: The current ``AttackedText`` on which the transformation was applied.
            original_text: The original ``AttackedText`` from which the attack started.
        """
        # Remove any occurences of current_text in transformed_texts
        transformed_texts = [
            t for t in transformed_texts if t.text != current_text.text
        ]
        # Populate cache with transformed_texts
        uncached_texts = []
        filtered_texts = []
        for transformed_text in transformed_texts:
            if (current_text, transformed_text) not in self.constraints_cache:
                uncached_texts.append(transformed_text)
            else:
                # promote transformed_text to the top of the LRU cache
                self.constraints_cache[
                    (current_text, transformed_text)
                ] = self.constraints_cache[(current_text, transformed_text)]
                if self.constraints_cache[(current_text, transformed_text)]:
                    filtered_texts.append(transformed_text)
        filtered_texts += self._filter_transformations_uncached(
            uncached_texts, current_text, original_text=original_text
        )
        # Sort transformations to ensure order is preserved between runs
        filtered_texts.sort(key=lambda t: t.text)
        return filtered_texts

    def _attack(self, initial_result):
        """Calls the ``SearchMethod`` to perturb the ``AttackedText`` stored in
        ``initial_result``.

        Args:
            initial_result: The initial ``GoalFunctionResult`` from which to perturb.

        Returns:
            A ``SuccessfulAttackResult``, ``FailedAttackResult``,
                or ``MaximizedAttackResult``.
        """
        final_result = self.search_method(initial_result)
        self.clear_cache()
        if final_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            return SuccessfulAttackResult(
                initial_result,
                final_result,
            )
        elif final_result.goal_status == GoalFunctionResultStatus.SEARCHING:
            return FailedAttackResult(
                initial_result,
                final_result,
            )
        elif final_result.goal_status == GoalFunctionResultStatus.MAXIMIZING:
            return MaximizedAttackResult(
                initial_result,
                final_result,
            )
        else:
            raise ValueError(f"Unrecognized goal status {final_result.goal_status}")

    def attack(self, example, ground_truth_output):
        """Attack a single example represented as ``AttackedText``
        Args:
            example (Union[AttackedText]): example to attack.
            ground_truth_output: ground truth output of ``example``.
        Returns:
            AttackResult
        """
        assert isinstance(
            example, AttackedText
        ), "`example` must be of type `AttackedText`."
        assert isinstance(
            ground_truth_output, (int, str)
        ), "`ground_truth_output` must either be `str` or `int`."
        goal_function_result, _ = self.goal_function.init_attack_example(
            example, ground_truth_output
        )
        if goal_function_result.goal_status == GoalFunctionResultStatus.SKIPPED:
            return SkippedAttackResult(goal_function_result)
        else:
            result = self._attack(goal_function_result)
            return result

    def __repr__(self):
        """Prints attack parameters in a human-readable string.

        Inspired by the readability of printing PyTorch nn.Modules:
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py
        """
        main_str = "Attack" + "("
        lines = []

        lines.append(utils.add_indent(f"(search_method): {self.search_method}", 2))
        # self.goal_function
        lines.append(utils.add_indent(f"(goal_function):  {self.goal_function}", 2))
        # self.transformation
        lines.append(utils.add_indent(f"(transformation):  {self.transformation}", 2))
        # self.constraints
        constraints_lines = []
        constraints = self.constraints + self.pre_transformation_constraints
        if len(constraints):
            for i, constraint in enumerate(constraints):
                constraints_lines.append(utils.add_indent(f"({i}): {constraint}", 2))
            constraints_str = utils.add_indent("\n" + "\n".join(constraints_lines), 2)
        else:
            constraints_str = "None"
        lines.append(utils.add_indent(f"(constraints): {constraints_str}", 2))
        # self.is_black_box
        lines.append(utils.add_indent(f"(is_black_box):  {self.is_black_box}", 2))
        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    def __getstate__(self):
        state = self.__dict__.copy()
        state["transformation_cache"] = None
        state["constraints_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.transformation_cache = lru.LRU(self.transformation_cache_size)
        self.constraints_cache = lru.LRU(self.constraint_cache_size)

    __str__ = __repr__
