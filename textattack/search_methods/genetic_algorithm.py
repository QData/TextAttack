"""
Reimplementation of search method from Generating Natural Language Adversarial Examples 
by Alzantot et. al
`<arxiv.org/abs/1804.07998>`_
`<github.com/nesl/nlp_adversarial_examples>`_
"""

from copy import deepcopy

import numpy as np
import torch

from textattack.search_methods import SearchMethod
from textattack.shared.validators import transformation_consists_of_word_swaps


class GeneticAlgorithm(SearchMethod):
    """
    Attacks a model with word substiutitions using a genetic algorithm.

    Args:
        pop_size (:obj:`int`, optional): The population size. Defauls to 20. 
        max_iters (:obj:`int`, optional): The maximum number of iterations to use. Defaults to 50. 
    """

    def __init__(
        self, pop_size=20, max_iters=50, temp=0.3, give_up_if_no_improvement=False
    ):
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.temp = temp
        self.give_up_if_no_improvement = give_up_if_no_improvement
        self.search_over = False

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
        orig_result, self.search_over = self.get_goal_results(
            [pop_member.attacked_text], self.correct_output
        )
        if self.search_over:
            return False
        new_x_results, self.search_over = self.get_goal_results(
            transformations, self.correct_output
        )
        new_x_scores = torch.Tensor([r.score for r in new_x_results])
        new_x_scores = new_x_scores - orig_result[0].score
        if len(new_x_scores) and new_x_scores.max() > 0:
            pop_member.attacked_text = transformations[new_x_scores.argmax()]
            return True
        return False

    def _perturb(self, pop_member):
        """
        Replaces a word in pop_member that has not been modified. 
        Args:
            pop_member: The population member being perturbed.
        """
        x_len = pop_member.neighbors_len.shape[0]
        neighbors_len = deepcopy(pop_member.neighbors_len)
        non_zero_indices = np.sum(np.sign(pop_member.neighbors_len))
        if non_zero_indices == 0:
            return
        iterations = 0
        while iterations < non_zero_indices and not self.search_over:
            w_select_probs = neighbors_len / np.sum(neighbors_len)
            rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
            if self._replace_at_index(pop_member, rand_idx):
                pop_member.neighbors_len[rand_idx] = 0
                break
            neighbors_len[rand_idx] = 0
            iterations += 1

    def _generate_population(self, neighbors_len, initial_result):
        """
        Generates a population of texts each with one word replaced
        Args:
            neighbors_len: A list of the number of candidate neighbors for each word.
            initial_result: The result to instantiate the population with
        Returns:
            The population.
        """
        pop = []
        for _ in range(self.pop_size):
            pop_member = PopulationMember(
                self.original_attacked_text, deepcopy(neighbors_len), initial_result
            )
            self._perturb(pop_member)
            pop.append(pop_member)
        return pop

    def _crossover(self, pop_member1, pop_member2):
        """
        Generates a crossover between pop_member1 and pop_member2.
        Args:
            pop_member1: The first population member.
            pop_member2: The second population member.
        Returns:
            A population member containing the crossover.
        """
        indices_to_replace = []
        words_to_replace = []
        x1_text = pop_member1.attacked_text
        x2_words = pop_member2.attacked_text.words
        new_neighbors_len = deepcopy(pop_member1.neighbors_len)
        for i in range(len(x1_text.words)):
            if np.random.uniform() < 0.5:
                indices_to_replace.append(i)
                words_to_replace.append(x2_words[i])
                new_neighbors_len[i] = pop_member2.neighbors_len[i]
        new_text = x1_text.replace_words_at_indices(
            indices_to_replace, words_to_replace
        )
        return PopulationMember(new_text, deepcopy(new_neighbors_len))

    def _get_neighbors_len(self, attacked_text):
        """
        Generates this neighbors_len list
        Args:
            attacked_text: The original text
        Returns:
            A list of number of candidate neighbors for each word
        """
        words = attacked_text.words
        neighbors_list = [[] for _ in range(len(words))]
        transformations = self.get_transformations(
            attacked_text, original_text=self.original_attacked_text
        )
        for transformed_text in transformations:
            diff_idx = attacked_text.first_word_diff_index(transformed_text)
            neighbors_list[diff_idx].append(transformed_text.words[diff_idx])
        neighbors_list = [np.array(x) for x in neighbors_list]
        neighbors_len = np.array([len(x) for x in neighbors_list])
        return neighbors_len

    def _perform_search(self, initial_result):
        self.original_attacked_text = initial_result.attacked_text
        self.correct_output = initial_result.output
        neighbors_len = self._get_neighbors_len(self.original_attacked_text)
        pop = self._generate_population(neighbors_len, initial_result)
        cur_score = initial_result.score
        for i in range(self.max_iters):
            pop_results, self.search_over = self.get_goal_results(
                [pm.attacked_text for pm in pop], self.correct_output
            )
            if self.search_over:
                if not len(pop_results):
                    return pop[0].result
                return max(pop_results, key=lambda x: x.score)
            for idx, result in enumerate(pop_results):
                pop[idx].result = pop_results[idx]
            pop = sorted(pop, key=lambda x: -x.result.score)

            pop_scores = torch.Tensor([r.score for r in pop_results])
            logits = ((-pop_scores) / self.temp).exp()
            select_probs = (logits / logits.sum()).cpu().numpy()

            if pop[0].result.succeeded:
                return pop[0].result

            if pop[0].result.score > cur_score:
                cur_score = pop[0].result.score
            elif self.give_up_if_no_improvement:
                break

            elite = [pop[0]]
            parent1_idx = np.random.choice(
                self.pop_size, size=self.pop_size - 1, p=select_probs
            )
            parent2_idx = np.random.choice(
                self.pop_size, size=self.pop_size - 1, p=select_probs
            )

            children = [
                self._crossover(pop[parent1_idx[idx]], pop[parent2_idx[idx]])
                for idx in range(self.pop_size - 1)
            ]
            for c in children:
                self._perturb(c)

            pop = elite + children

        return pop[0].result

    def check_transformation_compatibility(self, transformation):
        """
        The genetic algorithm is specifically designed for word substitutions.
        """
        return transformation_consists_of_word_swaps(transformation)

    def extra_repr_keys(self):
        return ["pop_size", "max_iters", "temp", "give_up_if_no_improvement"]


class PopulationMember:
    """
    A member of the population during the course of the genetic algorithm.
    
    Args:
        attacked_text: The ``AttackedText`` of the population member.
        neighbors_len: A list of the number of candidate neighbors list for each word.
    """

    def __init__(self, attacked_text, neighbors_len, result=None):
        self.attacked_text = attacked_text
        self.neighbors_len = neighbors_len
        self.result = result
