"""

Reimplementation of search method from Generating Natural Language Adversarial Examples
=========================================================================================

by Alzantot et. al `<arxiv.org/abs/1804.07998>`_  from `<github.com/nesl/nlp_adversarial_examples>`_
"""

import numpy as np

from textattack.search_methods import GeneticAlgorithm, PopulationMember


class AlzantotGeneticAlgorithm(GeneticAlgorithm):
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
        super().__init__(
            pop_size=pop_size,
            max_iters=max_iters,
            temp=temp,
            give_up_if_no_improvement=give_up_if_no_improvement,
            post_crossover_check=post_crossover_check,
            max_crossover_retries=max_crossover_retries,
        )

    def _modify_population_member(self, pop_member, new_text, new_result, word_idx):
        """Modify `pop_member` by returning a new copy with `new_text`,
        `new_result`, and `num_candidate_transformations` altered appropriately
        for given `word_idx`"""
        num_candidate_transformations = np.copy(
            pop_member.attributes["num_candidate_transformations"]
        )
        num_candidate_transformations[word_idx] = 0
        return PopulationMember(
            new_text,
            result=new_result,
            attributes={"num_candidate_transformations": num_candidate_transformations},
        )

    def _get_word_select_prob_weights(self, pop_member):
        """Get the attribute of `pop_member` that is used for determining
        probability of each word being selected for perturbation."""
        return pop_member.attributes["num_candidate_transformations"]

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
        num_candidate_transformations = np.copy(
            pop_member1.attributes["num_candidate_transformations"]
        )

        for i in range(pop_member1.num_words):
            if np.random.uniform() < 0.5:
                indices_to_replace.append(i)
                words_to_replace.append(pop_member2.words[i])
                num_candidate_transformations[i] = pop_member2.attributes[
                    "num_candidate_transformations"
                ][i]

        new_text = pop_member1.attacked_text.replace_words_at_indices(
            indices_to_replace, words_to_replace
        )
        return (
            new_text,
            {"num_candidate_transformations": num_candidate_transformations},
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
        num_candidate_transformations = np.zeros(len(words))
        transformed_texts = self.get_transformations(
            initial_result.attacked_text, original_text=initial_result.attacked_text
        )
        for transformed_text in transformed_texts:
            diff_idx = next(
                iter(transformed_text.attack_attrs["newly_modified_indices"])
            )
            num_candidate_transformations[diff_idx] += 1

        # Just b/c there are no replacements now doesn't mean we never want to select the word for perturbation
        # Therefore, we give small non-zero probability for words with no replacements
        # Epsilon is some small number to approximately assign small probability
        min_num_candidates = np.amin(num_candidate_transformations)
        epsilon = max(1, int(min_num_candidates * 0.1))
        for i in range(len(num_candidate_transformations)):
            num_candidate_transformations[i] = max(
                num_candidate_transformations[i], epsilon
            )

        population = []
        for _ in range(pop_size):
            pop_member = PopulationMember(
                initial_result.attacked_text,
                initial_result,
                attributes={
                    "num_candidate_transformations": np.copy(
                        num_candidate_transformations
                    )
                },
            )
            # Perturb `pop_member` in-place
            pop_member = self._perturb(pop_member, initial_result)
            population.append(pop_member)

        return population
