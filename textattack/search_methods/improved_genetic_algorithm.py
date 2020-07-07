"""
Reimplementation of search method from Natural Language Adversarial Attacks and Defenses in Word Level 
by Wang et al.
`<arxiv.org/abs/1909.06723>`_
`<github.com/CodeforAnonymousPaper/SEM>`_
"""

from copy import deepcopy

import numpy as np
import torch

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import transformation_consists_of_word_swaps


class ImprovedGeneticAlgorithm(SearchMethod):
    """
    Attacks a model with word substiutitions using a genetic algorithm.

    Args:
        max_pop_size (int): The maximum population size. Defaults to 20. 
        max_iters (int): The maximum number of iterations to use. Defaults to 50.
        max_replaced_times (int): The maximum number of times each word in the same position can be replaced at. Defaults to 5.
        give_up_if_no_improvement (bool): If True, stop the search early if no candidate that improves the score is found.
        max_crossover_retries (int): Maximum number of crossover retries if resulting child fails to pass the constraints.
            Setting it to 0 means we immediately take one of the parents at random as the child.
    """

    def __init__(
        self,
        max_pop_size=20,
        max_iters=50,
        max_replaced_times=5,
        give_up_if_no_improvement=False,
        max_crossover_retries=20,
    ):
        self.max_iters = max_iters
        self.max_pop_size = max_pop_size
        self.max_replaced_times = max_replaced_times
        self.give_up_if_no_improvement = give_up_if_no_improvement
        self.max_crossover_retries = max_crossover_retries
        # flag to indicate if it is the improved genetic algorithm proposed in Natural Language Adversarial Attacks and Defenses in Word Level by Wang et al.

        # internal flag to indicate if search should end immediately
        self._search_over = False



    def _replace_at_index(self, pop_member, idx):
        """
        Select the best replacement for word at position (idx) 
        in (pop_member) to maximize score.

        Args:
            pop_member: The population member being perturbed.
            idx: The index at which to replace a word.

        Returns:
            Whether a replacement which increased the score was found.
        """
        transformations = self.get_transformations(
            pop_member.attacked_text,
            original_text=self.original_attacked_text,
            indices_to_modify=[idx],
        )
        if not len(transformations):
            return False
        orig_result, self._search_over = self.get_goal_results(
            [pop_member.attacked_text]
        )
        if self._search_over:
            return False
        new_x_results, self._search_over = self.get_goal_results(transformations)
        new_x_scores = torch.Tensor([r.score for r in new_x_results])
        new_x_scores = new_x_scores - orig_result[0].score
        if len(new_x_scores) and new_x_scores.max() > 0:
            max_idx = new_x_scores.argmax()
            pop_member.attacked_text = transformations[max_idx]
            pop_member.replaced_times[idx] = -1
            pop_member.results = new_x_results[max_idx]
            return True
        return False

    def _mutate(self, pop_member):
        """
        Replaces a word in pop_member that has not been modified. 
        Args:
            pop_member: The population member being perturbed.
        """
        x_len = pop_member.replaced_times.shape[0]
        iterations = 0
        while iterations < x_len and not self._search_over:
            rand_idx = np.random.randint(0, x_len)
            if pop_member.replaced_times[rand_idx] == 0:
                continue
            if self._replace_at_index(pop_member, rand_idx):
                break
            iterations += 1

    def _initialize_population(self, initial_result):
        """
        Initialize a population of texts by replacing each word with its optimal synonym.
        Args:
            initial_result (GoalFunctionResult): The result to instantiate the population with.
        Returns:
            The population.
        """
        words = initial_result.attacked_text.words
        replaced_times = np.array([self.max_replaced_times for _ in range(len(words))])
        population = []
        for idx in range(len(words)):
            pop_member = PopulationMember(
                self.original_attacked_text, np.copy(replaced_times), initial_result
            )
            self._replace_at_index(pop_member, idx)
            population.append(pop_member)
        return population

    def _crossover(self, pop_member1, pop_member2):
        """
        Generates a crossover between pop_member1 and pop_member2.
        If the child fails to satisfy the constraits, we re-try crossover for a fix number of times,
        before taking one of the parents at random as the resulting child.
        Args:
            pop_member1 (PopulationMember): The first population member.
            pop_member2 (PopulationMember): The second population member.
        Returns:
            A population member containing the crossover.
        """
        x1_text = pop_member1.attacked_text
        x2_text = pop_member2.attacked_text

        num_tries = 0
        passed_constraints = False
        while num_tries < self.max_crossover_retries + 1:
            new_replaced_times = np.copy(pop_member1.replaced_times)
            crossover_point = np.random.randint(0, len(new_replaced_times))
            indices_to_replace = []
            words_to_replace = []
            for i in range(crossover_point, len(new_replaced_times)):
                indices_to_replace.append(i)
                words_to_replace.append(x2_text.words[i])
                new_replaced_times[i] = pop_member2.replaced_times[i]
            new_text = x1_text.replace_words_at_indices(
                indices_to_replace, words_to_replace
            )
            if "last_transformation" in x1_text.attack_attrs:
                new_text.attack_attrs["last_transformation"] = x1_text.attack_attrs[
                    "last_transformation"
                ]
                filtered = self.filter_transformations(
                    [new_text], x1_text, original_text=self.original_attacked_text
                )
            elif "last_transformation" in x2_text.attack_attrs:
                new_text.attack_attrs["last_transformation"] = x2_text.attack_attrs[
                    "last_transformation"
                ]
                filtered = self.filter_transformations(
                    [new_text], x1_text, original_text=self.original_attacked_text
                )
            else:
                # In this case, neither x_1 nor x_2 has been transformed,
                # meaning that new_text == original_text
                filtered = [new_text]

            if filtered:
                new_text = filtered[0]
                passed_constraints = True
                break

            num_tries += 1

        if not passed_constraints:
            # If we cannot find a child that passes the constraints,
            # we just randomly pick one of the parents to be the child for the next iteration.
            new_text = (
                pop_member1.attacked_text
                if np.random.uniform() < 0.5
                else pop_member2.attacked_text
            )

        new_results, self._search_over = self.get_goal_results([new_text])

        return PopulationMember(new_text, new_replaced_times, new_results[0])

    def _perform_search(self, initial_result):
        self.original_attacked_text = initial_result.attacked_text
        population = self._initialize_population(initial_result)
        current_score = initial_result.score
        for i in range(self.max_iters):
            population = sorted(population, key=lambda x: x.result.score, reverse=True)
            pop_size = len(population)
            if pop_size > self.max_pop_size:
                population = population[: self.max_pop_size]
                pop_size = self.max_pop_size

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

            elite = [population[0]]
            parent1_idx = np.random.choice(pop_size, size=pop_size - 1)
            parent2_idx = np.random.choice(pop_size, size=pop_size - 1)

            children = []
            for idx in range(pop_size - 1):
                child = self._crossover(
                    population[parent1_idx[idx]], population[parent2_idx[idx]]
                )
                if self._search_over:
                    break

                self._mutate(child)
                children.append(child)

                if self._search_over:
                    break

            population = elite + children

        return population[0].result

    def check_transformation_compatibility(self, transformation):
        """
        The genetic algorithm is specifically designed for word substitutions.
        """
        return transformation_consists_of_word_swaps(transformation)

    def extra_repr_keys(self):
        return [
            "max_pop_size",
            "max_iters",
            "max_replaced_times",
            "give_up_if_no_improvement",
        ]


class PopulationMember:
    """
    A member of the population during the course of the improved genetic algorithm.
    
    Args:
        attacked_text: The ``AttackedText`` of the population member.
        replaced_times: A list of the number of candidate neighbors list for each word.
    """

    def __init__(self, attacked_text, replaced_times, result=None):
        self.attacked_text = attacked_text
        self.replaced_times = replaced_times
        self.result = result
