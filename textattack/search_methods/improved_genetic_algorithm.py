"""

Reimplementation of search method from Xiaosen Wang, Hao Jin, Kun He (2019).
=========================================================================================


Natural Language Adversarial Attack and Defense in Word Level.
http://arxiv.org/abs/1909.06723
"""

import numpy as np

from textattack.search_methods import GeneticAlgorithm, PopulationMember


class ImprovedGeneticAlgorithm(GeneticAlgorithm):
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
        max_replace_times_per_index (int):  The maximum times words at the same index can be replaced in improved genetic algorithm.
    """

    def __init__(
        self,
        pop_size=60,
        max_iters=20,
        temp=0.3,
        give_up_if_no_improvement=False,
        post_crossover_check=True,
        max_crossover_retries=20,
        max_replace_times_per_index=5,
    ):
        super().__init__(
            pop_size=pop_size,
            max_iters=max_iters,
            temp=temp,
            give_up_if_no_improvement=give_up_if_no_improvement,
            post_crossover_check=post_crossover_check,
            max_crossover_retries=max_crossover_retries,
        )

        self.max_replace_times_per_index = max_replace_times_per_index

    def _modify_population_member(self, pop_member, new_text, new_result, word_idx):
        """Modify `pop_member` by returning a new copy with `new_text`,
        `new_result`, and `num_replacements_left` altered appropriately for
        given `word_idx`"""
        num_replacements_left = np.copy(pop_member.attributes["num_replacements_left"])
        num_replacements_left[word_idx] -= 1
        return PopulationMember(
            new_text,
            result=new_result,
            attributes={"num_replacements_left": num_replacements_left},
        )

    def _get_word_select_prob_weights(self, pop_member):
        """Get the attribute of `pop_member` that is used for determining
        probability of each word being selected for perturbation."""
        return pop_member.attributes["num_replacements_left"]

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
        indices_to_replace = []
        words_to_replace = []
        num_replacements_left = np.copy(pop_member1.attributes["num_replacements_left"])

        # To better simulate the reproduction and biological crossover,
        # IGA randomly cut the text from two parents and concat two fragments into a new text
        # rather than randomly choose a word of each position from the two parents.
        crossover_point = np.random.randint(0, pop_member1.num_words)
        for i in range(crossover_point, pop_member1.num_words):
            indices_to_replace.append(i)
            words_to_replace.append(pop_member2.words[i])
            num_replacements_left[i] = pop_member2.attributes["num_replacements_left"][
                i
            ]

        new_text = pop_member1.attacked_text.replace_words_at_indices(
            indices_to_replace, words_to_replace
        )
        return new_text, {"num_replacements_left": num_replacements_left}

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
        # For IGA, `num_replacements_left` represents the number of times the word at each index can be modified
        num_replacements_left = np.array(
            [self.max_replace_times_per_index] * len(words)
        )
        population = []

        # IGA initializes the first population by replacing each word by its optimal synonym
        for idx in range(len(words)):
            pop_member = PopulationMember(
                initial_result.attacked_text,
                initial_result,
                attributes={"num_replacements_left": np.copy(num_replacements_left)},
            )
            pop_member = self._perturb(pop_member, initial_result, index=idx)
            population.append(pop_member)

        return population[:pop_size]

    def extra_repr_keys(self):
        return super().extra_repr_keys() + ["max_replace_times_per_index"]
