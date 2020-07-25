"""Reimplementation of search method from Generating Natural Language
Adversarial Examples by Alzantot et.

al `<arxiv.org/abs/1804.07998>`_
`<github.com/nesl/nlp_adversarial_examples>`_
"""

# from copy import deepcopy

import numpy as np
import torch

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import PopulationBasedSearch, PopulationMember
from textattack.shared.validators import transformation_consists_of_word_swaps


class GeneticAlgorithm(PopulationBasedSearch):
    """Attacks a model with word substiutitions using a genetic algorithm.

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

    def _perturb(self, pop_member, original_result):
        """Perturb `pop_member` in-place.

        Replaces a word at a random in `pop_member` with replacement word that maximizes increase in score.
        Args:
            pop_member (PopulationMember): The population member being perturbed.
            original_result (GoalFunctionResult): Result of original sample being attacked
        Returns:
            `True` if perturbation occurred. `False` if not.
        """
        num_words = pop_member.num_replacements_per_word.shape[0]
        num_replacements_per_word = np.copy(pop_member.num_replacements_per_word)
        non_zero_indices = np.count_nonzero(num_replacements_per_word)
        if non_zero_indices == 0:
            return False
        iterations = 0
        while iterations < non_zero_indices:
            w_select_probs = num_replacements_per_word / np.sum(
                num_replacements_per_word
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

            if self._search_over:
                break

            diff_scores = (
                torch.Tensor([r.score for r in new_results]) - pop_member.result.score
            )
            if len(diff_scores) and diff_scores.max() > 0:
                idx_with_max_score = diff_scores.argmax()
                pop_member.attacked_text = transformed_texts[idx_with_max_score]
                pop_member.results = new_results[idx_with_max_score]
                pop_member.num_replacements_per_word[idx] = 0
                return True

            num_replacements_per_word[idx] = 0
            iterations += 1
        return False

    def _crossover(self, pop_member1, pop_member2, original_text):
        """Generates a crossover between pop_member1 and pop_member2.

        If the child fails to satisfy the constraits, we re-try crossover for a fix number of times,
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
        x2_words = x2_text.words

        num_tries = 0
        passed_constraints = False
        while num_tries < self.max_crossover_retries + 1:
            indices_to_replace = []
            words_to_replace = []
            num_replacements_per_word = np.copy(pop_member1.num_replacements_per_word)

            for i in range(len(x1_text.words)):
                if np.random.uniform() < 0.5:
                    indices_to_replace.append(i)
                    words_to_replace.append(x2_words[i])
                    num_replacements_per_word[
                        i
                    ] = pop_member2.num_replacements_per_word[i]

            new_text = x1_text.replace_words_at_indices(
                indices_to_replace, words_to_replace
            )
            indices_to_replace = set(indices_to_replace)
            new_text.attack_attrs["modified_indices"] = (
                x1_text.attack_attrs["modified_indices"] - indices_to_replace
            ) | (x2_text.attack_attrs["modified_indices"] & indices_to_replace)

            if "last_transformation" in x1_text.attack_attrs:
                new_text.attack_attrs["last_transformation"] = x1_text.attack_attrs[
                    "last_transformation"
                ]
            elif "last_transformation" in x2_text.attack_attrs:
                new_text.attack_attrs["last_transformation"] = x2_text.attack_attrs[
                    "last_transformation"
                ]

            if not self.post_crossover_check or (
                new_text.text == x1_text.text or new_text.text == x2_text.text
            ):
                break

            if "last_transformation" in new_text.attack_attrs:
                previous_text = (
                    x1_text
                    if "last_transformation" in x1_text.attack_attrs
                    else x2_text
                )
                passed_constraints = self._check_constraints(
                    new_text, previous_text, original_text=original_text
                )
            else:
                passed_constraints = True

            if passed_constraints:
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
                new_text,
                new_results[0],
                num_replacements_per_word=num_replacements_per_word,
            )

    def _initialize_population(self, initial_result, pop_size):
        """
        Initialize a population of size `pop_size` with `initial_result`
        Args:
            initial_result (GoalFunctionResult): Original text
            pop_size (int): size of population
        Returns:
            population as `list[PopulationMember]`
        """
        words = initial_result.attacked_text.words
        num_replacements_per_word = np.zeros(len(words))
        transformed_texts = self.get_transformations(
            initial_result.attacked_text, original_text=initial_result.attacked_text
        )
        for transformed_text in transformed_texts:
            diff_idx = next(
                iter(transformed_text.attack_attrs["newly_modified_indices"])
            )
            num_replacements_per_word[diff_idx] += 1

        # Just b/c there are no replacements now doesn't mean we never want to select the word for perturbation
        # Therefore, we give small non-zero probability for words with no replacements
        # Epsilon is some small number to approximately assign small probability
        min_num_candidates = np.amin(num_replacements_per_word)
        epsilon = max(1, int(min_num_candidates * 0.1))
        for i in range(len(num_replacements_per_word)):
            num_replacements_per_word[i] = max(num_replacements_per_word[i], epsilon)

        population = []
        for _ in range(pop_size):
            pop_member = PopulationMember(
                initial_result.attacked_text,
                initial_result,
                num_replacements_per_word=np.copy(num_replacements_per_word),
            )
            # Perturb `pop_member` in-place
            self._perturb(pop_member, initial_result)
            population.append(pop_member)

        return population

    def _perform_search(self, initial_result):
        self._search_over = False
        population = self._initialize_population(initial_result, self.pop_size)
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

            parent1_idx = np.random.choice(
                self.pop_size, size=self.pop_size - 1, p=select_probs
            )
            parent2_idx = np.random.choice(
                self.pop_size, size=self.pop_size - 1, p=select_probs
            )

            children = []
            for idx in range(self.pop_size - 1):
                child = self._crossover(
                    population[parent1_idx[idx]],
                    population[parent2_idx[idx]],
                    initial_result.attacked_text,
                )
                if self._search_over:
                    break

                self._perturb(child, initial_result)
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

    def extra_repr_keys(self):
        return [
            "pop_size",
            "max_iters",
            "temp",
            "give_up_if_no_improvement",
            "post_crossover_check",
            "max_crossover_retries",
        ]
