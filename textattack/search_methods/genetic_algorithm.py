"""
Genetic Algorithm Word Swap
====================================
"""
from abc import ABC, abstractmethod

import numpy as np
import torch

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import PopulationBasedSearch, PopulationMember
from textattack.shared.validators import transformation_consists_of_word_swaps


class GeneticAlgorithm(PopulationBasedSearch, ABC):
    """Base class for attacking a model with word substiutitions using a
    genetic algorithm.

    Args:
        pop_size (int): The population size. Defaults to 20.
        max_iters (int): The maximum number of iterations to use. Defaults to 50.
        temp (float): Temperature for softmax function used to normalize probability dist when sampling parents.
            Higher temperature increases the sensitivity to lower probability candidates.
        give_up_if_no_improvement (bool): If True, stop the search early if no candidate that improves the score is found.
        post_crossover_check (bool): If True, check if child produced from crossover step passes the constraints.
        max_crossover_retries (int): Maximum number of crossover retries if resulting child fails to pass the constraints.
            Applied only when `post_crossover_check` is set to `True`.
            Setting it to 0 means we immediately take one of the parents at random as the child upon failure.
    """

    def __init__(
        self,
        pop_size=60,
        max_iters=20,
        temp=0.3,
        give_up_if_no_improvement=False,
        post_crossover_check=True,
        max_crossover_retries=20,
    ):
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.temp = temp
        self.give_up_if_no_improvement = give_up_if_no_improvement
        self.post_crossover_check = post_crossover_check
        self.max_crossover_retries = max_crossover_retries

        # internal flag to indicate if search should end immediately
        self._search_over = False

    @abstractmethod
    def _modify_population_member(self, pop_member, new_text, new_result, word_idx):
        """Modify `pop_member` by returning a new copy with `new_text`,
        `new_result`, and, `attributes` altered appropriately for given
        `word_idx`"""
        raise NotImplementedError()

    @abstractmethod
    def _get_word_select_prob_weights(self, pop_member):
        """Get the attribute of `pop_member` that is used for determining
        probability of each word being selected for perturbation."""
        raise NotImplementedError

    def _perturb(self, pop_member, original_result, index=None):
        """Perturb `pop_member` and return it. Replaces a word at a random
        (unless `index` is specified) in `pop_member`.

        Args:
            pop_member (PopulationMember): The population member being perturbed.
            original_result (GoalFunctionResult): Result of original sample being attacked
            index (int): Index of word to perturb.
        Returns:
            Perturbed `PopulationMember`
        """
        num_words = pop_member.attacked_text.num_words
        # `word_select_prob_weights` is a list of values used for sampling one word to transform
        word_select_prob_weights = np.copy(
            self._get_word_select_prob_weights(pop_member)
        )
        non_zero_indices = np.count_nonzero(word_select_prob_weights)
        if non_zero_indices == 0:
            return pop_member
        iterations = 0
        while iterations < non_zero_indices:
            if index:
                idx = index
            else:
                w_select_probs = word_select_prob_weights / np.sum(
                    word_select_prob_weights
                )
                idx = np.random.choice(num_words, 1, p=w_select_probs)[0]

            transformed_texts = self.get_transformations(
                pop_member.attacked_text,
                original_text=original_result.attacked_text,
                indices_to_modify=[idx],
            )

            if not len(transformed_texts):
                iterations += 1
                continue

            new_results, self._search_over = self.get_goal_results(transformed_texts)

            diff_scores = (
                torch.Tensor([r.score for r in new_results]) - pop_member.result.score
            )
            if len(diff_scores) and diff_scores.max() > 0:
                idx_with_max_score = diff_scores.argmax()
                pop_member = self._modify_population_member(
                    pop_member,
                    transformed_texts[idx_with_max_score],
                    new_results[idx_with_max_score],
                    idx,
                )
                return pop_member

            word_select_prob_weights[idx] = 0
            iterations += 1

            if self._search_over:
                break

        return pop_member

    @abstractmethod
    def _crossover_operation(self, pop_member1, pop_member2):
        """Actual operation that takes `pop_member1` text and `pop_member2`
        text and mixes the two to generate crossover between `pop_member1` and
        `pop_member2`.

        Args:
            pop_member1 (PopulationMember): The first population member.
            pop_member2 (PopulationMember): The second population member.
        Returns:
            Tuple of `AttackedText` and a dictionary of attributes.
        """
        raise NotImplementedError()

    def _post_crossover_check(
        self, new_text, parent_text1, parent_text2, original_text
    ):
        """Check if `new_text` that has been produced by performing crossover
        between `parent_text1` and `parent_text2` aligns with the constraints.

        Args:
            new_text (AttackedText): Text produced by crossover operation
            parent_text1 (AttackedText): Parent text of `new_text`
            parent_text2 (AttackedText): Second parent text of `new_text`
            original_text (AttackedText): Original text
        Returns:
            `True` if `new_text` meets the constraints. If otherwise, return `False`.
        """
        if "last_transformation" in new_text.attack_attrs:
            previous_text = (
                parent_text1
                if "last_transformation" in parent_text1.attack_attrs
                else parent_text2
            )
            passed_constraints = self._check_constraints(
                new_text, previous_text, original_text=original_text
            )
            return passed_constraints
        else:
            # `new_text` has not been actually transformed, so return True
            return True

    def _crossover(self, pop_member1, pop_member2, original_text):
        """Generates a crossover between pop_member1 and pop_member2.

        If the child fails to satisfy the constraints, we re-try crossover for a fix number of times,
        before taking one of the parents at random as the resulting child.
        Args:
            pop_member1 (PopulationMember): The first population member.
            pop_member2 (PopulationMember): The second population member.
            original_text (AttackedText): Original text
        Returns:
            A population member containing the crossover.
        """
        x1_text = pop_member1.attacked_text
        x2_text = pop_member2.attacked_text

        num_tries = 0
        passed_constraints = False
        while num_tries < self.max_crossover_retries + 1:
            new_text, attributes = self._crossover_operation(pop_member1, pop_member2)

            replaced_indices = new_text.attack_attrs["newly_modified_indices"]
            new_text.attack_attrs["modified_indices"] = (
                x1_text.attack_attrs["modified_indices"] - replaced_indices
            ) | (x2_text.attack_attrs["modified_indices"] & replaced_indices)

            if "last_transformation" in x1_text.attack_attrs:
                new_text.attack_attrs["last_transformation"] = x1_text.attack_attrs[
                    "last_transformation"
                ]
            elif "last_transformation" in x2_text.attack_attrs:
                new_text.attack_attrs["last_transformation"] = x2_text.attack_attrs[
                    "last_transformation"
                ]

            if self.post_crossover_check:
                passed_constraints = self._post_crossover_check(
                    new_text, x1_text, x2_text, original_text
                )

            if not self.post_crossover_check or passed_constraints:
                break

            num_tries += 1

        if self.post_crossover_check and not passed_constraints:
            # If we cannot find a child that passes the constraints,
            # we just randomly pick one of the parents to be the child for the next iteration.
            pop_mem = pop_member1 if np.random.uniform() < 0.5 else pop_member2
            return pop_mem
        else:
            new_results, self._search_over = self.get_goal_results([new_text])
            return PopulationMember(
                new_text, result=new_results[0], attributes=attributes
            )

    @abstractmethod
    def _initialize_population(self, initial_result, pop_size):
        """
        Initialize a population of size `pop_size` with `initial_result`
        Args:
            initial_result (GoalFunctionResult): Original text
            pop_size (int): size of population
        Returns:
            population as `list[PopulationMember]`
        """
        raise NotImplementedError()

    def perform_search(self, initial_result):
        self._search_over = False
        population = self._initialize_population(initial_result, self.pop_size)
        pop_size = len(population)
        current_score = initial_result.score

        for i in range(self.max_iters):
            population = sorted(population, key=lambda x: x.result.score, reverse=True)

            if (
                self._search_over
                or population[0].result.goal_status
                == GoalFunctionResultStatus.SUCCEEDED
            ):
                break

            if population[0].result.score > current_score:
                current_score = population[0].result.score
            elif self.give_up_if_no_improvement:
                break

            pop_scores = torch.Tensor([pm.result.score for pm in population])
            logits = ((-pop_scores) / self.temp).exp()
            select_probs = (logits / logits.sum()).cpu().numpy()

            parent1_idx = np.random.choice(pop_size, size=pop_size - 1, p=select_probs)
            parent2_idx = np.random.choice(pop_size, size=pop_size - 1, p=select_probs)

            children = []
            for idx in range(pop_size - 1):
                child = self._crossover(
                    population[parent1_idx[idx]],
                    population[parent2_idx[idx]],
                    initial_result.attacked_text,
                )
                if self._search_over:
                    break

                child = self._perturb(child, initial_result)
                children.append(child)

                # We need two `search_over` checks b/c value might change both in
                # `crossover` method and `perturb` method.
                if self._search_over:
                    break

            population = [population[0]] + children

        return population[0].result

    def check_transformation_compatibility(self, transformation):
        """The genetic algorithm is specifically designed for word
        substitutions."""
        return transformation_consists_of_word_swaps(transformation)

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return [
            "pop_size",
            "max_iters",
            "temp",
            "give_up_if_no_improvement",
            "post_crossover_check",
            "max_crossover_retries",
        ]
