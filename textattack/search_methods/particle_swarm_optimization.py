"""

Particle Swarm Optimization
====================================

Reimplementation of search method from Word-level Textual Adversarial
Attacking as Combinatorial Optimization by Zang et.

al
`<https://www.aclweb.org/anthology/2020.acl-main.540.pdf>`_
`<https://github.com/thunlp/SememePSO-Attack>`_
"""
import copy

import numpy as np

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import PopulationBasedSearch, PopulationMember
from textattack.shared import utils
from textattack.shared.validators import transformation_consists_of_word_swaps


class ParticleSwarmOptimization(PopulationBasedSearch):
    """Attacks a model with word substiutitions using a Particle Swarm
    Optimization (PSO) algorithm. Some key hyper-parameters are setup according
    to the original paper:

    "We adjust PSO on the validation set of SST and set ω_1 as 0.8 and ω_2 as 0.2.
    We set the max velocity of the particles V_{max} to 3, which means the changing
    probability of the particles ranges from 0.047 (sigmoid(-3)) to 0.953 (sigmoid(3))."

    Args:
        pop_size (:obj:`int`, optional): The population size. Defaults to 60.
        max_iters (:obj:`int`, optional): The maximum number of iterations to use. Defaults to 20.
        post_turn_check (:obj:`bool`, optional): If `True`, check if new position reached by moving passes the constraints. Defaults to `True`
        max_turn_retries (:obj:`bool`, optional): Maximum number of movement retries if new position after turning fails to pass the constraints.
            Applied only when `post_movement_check` is set to `True`.
            Setting it to 0 means we immediately take the old position as the new position upon failure.
    """

    def __init__(
        self, pop_size=60, max_iters=20, post_turn_check=True, max_turn_retries=20
    ):
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.post_turn_check = post_turn_check
        self.max_turn_retries = 20

        self._search_over = False
        self.omega_1 = 0.8
        self.omega_2 = 0.2
        self.c1_origin = 0.8
        self.c2_origin = 0.2
        self.v_max = 3.0

    def _perturb(self, pop_member, original_result):
        """Perturb `pop_member` in-place.

        Replaces a word at a random in `pop_member` with replacement word that maximizes increase in score.
        Args:
            pop_member (PopulationMember): The population member being perturbed.
            original_result (GoalFunctionResult): Result of original sample being attacked
        Returns:
            `True` if perturbation occured. `False` if not.
        """
        # TODO: Below is very slow and is the main cause behind memory build up + slowness
        best_neighbors, prob_list = self._get_best_neighbors(
            pop_member.result, original_result
        )
        random_result = np.random.choice(best_neighbors, 1, p=prob_list)[0]

        if random_result == pop_member.result:
            return False
        else:
            pop_member.attacked_text = random_result.attacked_text
            pop_member.result = random_result
            return True

    def _equal(self, a, b):
        return -self.v_max if a == b else self.v_max

    def _turn(self, source_text, target_text, prob, original_text):
        """
        Based on given probabilities, "move" to `target_text` from `source_text`
        Args:
            source_text (PopulationMember): Text we start from.
            target_text (PopulationMember): Text we want to move to.
            prob (np.array[float]): Turn probability for each word.
            original_text (AttackedText): Original text for constraint check if `self.post_turn_check=True`.
        Returns:
            New `Position` that we moved to (or if we fail to move, same as `source_text`)
        """
        assert len(source_text.words) == len(
            target_text.words
        ), "Word length mismatch for turn operation."
        assert len(source_text.words) == len(
            prob
        ), "Length mismatch for words and probability list."
        len_x = len(source_text.words)

        num_tries = 0
        passed_constraints = False
        while num_tries < self.max_turn_retries + 1:
            indices_to_replace = []
            words_to_replace = []
            for i in range(len_x):
                if np.random.uniform() < prob[i]:
                    indices_to_replace.append(i)
                    words_to_replace.append(target_text.words[i])
            new_text = source_text.attacked_text.replace_words_at_indices(
                indices_to_replace, words_to_replace
            )
            indices_to_replace = set(indices_to_replace)
            new_text.attack_attrs["modified_indices"] = (
                source_text.attacked_text.attack_attrs["modified_indices"]
                - indices_to_replace
            ) | (
                target_text.attacked_text.attack_attrs["modified_indices"]
                & indices_to_replace
            )
            if "last_transformation" in source_text.attacked_text.attack_attrs:
                new_text.attack_attrs[
                    "last_transformation"
                ] = source_text.attacked_text.attack_attrs["last_transformation"]

            if not self.post_turn_check or (new_text.words == source_text.words):
                break

            if "last_transformation" in new_text.attack_attrs:
                passed_constraints = self._check_constraints(
                    new_text, source_text.attacked_text, original_text=original_text
                )
            else:
                passed_constraints = True

            if passed_constraints:
                break

            num_tries += 1

        if self.post_turn_check and not passed_constraints:
            # If we cannot find a turn that passes the constraints, we do not move.
            return source_text
        else:
            return PopulationMember(new_text)

    def _get_best_neighbors(self, current_result, original_result):
        """For given current text, find its neighboring texts that yields
        maximum improvement (in goal function score) for each word.

        Args:
            current_result (GoalFunctionResult): `GoalFunctionResult` of current text
            original_result (GoalFunctionResult): `GoalFunctionResult` of original text.
        Returns:
            best_neighbors (list[GoalFunctionResult]): Best neighboring text for each word
            prob_list (list[float]): discrete probablity distribution for sampling a neighbor from `best_neighbors`
        """
        current_text = current_result.attacked_text
        neighbors_list = [[] for _ in range(len(current_text.words))]
        transformed_texts = self.get_transformations(
            current_text, original_text=original_result.attacked_text
        )
        for transformed_text in transformed_texts:
            diff_idx = next(
                iter(transformed_text.attack_attrs["newly_modified_indices"])
            )
            neighbors_list[diff_idx].append(transformed_text)

        best_neighbors = []
        score_list = []
        for i in range(len(neighbors_list)):
            if not neighbors_list[i]:
                best_neighbors.append(current_result)
                score_list.append(0)
                continue

            neighbor_results, self._search_over = self.get_goal_results(
                neighbors_list[i]
            )
            if not len(neighbor_results):
                best_neighbors.append(current_result)
                score_list.append(0)
            else:
                neighbor_scores = np.array([r.score for r in neighbor_results])
                score_diff = neighbor_scores - current_result.score
                best_idx = np.argmax(neighbor_scores)
                best_neighbors.append(neighbor_results[best_idx])
                score_list.append(score_diff[best_idx])

        prob_list = normalize(score_list)

        return best_neighbors, prob_list

    def _initialize_population(self, initial_result, pop_size):
        """
        Initialize a population of size `pop_size` with `initial_result`
        Args:
            initial_result (GoalFunctionResult): Original text
            pop_size (int): size of population
        Returns:
            population as `list[PopulationMember]`
        """
        best_neighbors, prob_list = self._get_best_neighbors(
            initial_result, initial_result
        )
        population = []
        for _ in range(pop_size):
            # Mutation step
            random_result = np.random.choice(best_neighbors, 1, p=prob_list)[0]
            population.append(
                PopulationMember(random_result.attacked_text, random_result)
            )
        return population

    def _perform_search(self, initial_result):
        self._search_over = False
        population = self._initialize_population(initial_result, self.pop_size)
        # Initialize  up velocities of each word for each population
        v_init = np.random.uniform(-self.v_max, self.v_max, self.pop_size)
        velocities = np.array(
            [
                [v_init[t] for _ in range(initial_result.attacked_text.num_words)]
                for t in range(self.pop_size)
            ]
        )

        global_elite = max(population, key=lambda x: x.score)
        if (
            self._search_over
            or global_elite.result.goal_status == GoalFunctionResultStatus.SUCCEEDED
        ):
            return global_elite.result

        local_elites = copy.copy(population)

        # start iterations
        for i in range(self.max_iters):
            omega = (self.omega_1 - self.omega_2) * (
                self.max_iters - i
            ) / self.max_iters + self.omega_2
            C1 = self.c1_origin - i / self.max_iters * (self.c1_origin - self.c2_origin)
            C2 = self.c2_origin + i / self.max_iters * (self.c1_origin - self.c2_origin)
            P1 = C1
            P2 = C2

            for k in range(len(population)):
                # calculate the probability of turning each word
                pop_mem_words = population[k].words
                local_elite_words = local_elites[k].words
                assert len(pop_mem_words) == len(
                    local_elite_words
                ), "PSO word length mismatch!"

                for d in range(len(pop_mem_words)):
                    velocities[k][d] = omega * velocities[k][d] + (1 - omega) * (
                        self._equal(pop_mem_words[d], local_elite_words[d])
                        + self._equal(pop_mem_words[d], global_elite.words[d])
                    )
                turn_prob = utils.sigmoid(velocities[k])

                if np.random.uniform() < P1:
                    # Move towards local elite
                    population[k] = self._turn(
                        local_elites[k],
                        population[k],
                        turn_prob,
                        initial_result.attacked_text,
                    )

                if np.random.uniform() < P2:
                    # Move towards global elite
                    population[k] = self._turn(
                        global_elite,
                        population[k],
                        turn_prob,
                        initial_result.attacked_text,
                    )

            # Check if there is any successful attack in the current population
            pop_results, self._search_over = self.get_goal_results(
                [p.attacked_text for p in population]
            )
            if self._search_over:
                # if `get_goal_results` gets cut short by query budget, resize population
                population = population[: len(pop_results)]
            for k in range(len(pop_results)):
                population[k].result = pop_results[k]

            top_member = max(population, key=lambda x: x.score)
            if (
                self._search_over
                or top_member.result.goal_status == GoalFunctionResultStatus.SUCCEEDED
            ):
                return top_member.result

            # Mutation based on the current change rate
            for k in range(len(population)):
                change_ratio = initial_result.attacked_text.words_diff_ratio(
                    population[k].attacked_text
                )
                # Referred from the original source code
                p_change = 1 - 2 * change_ratio
                if np.random.uniform() < p_change:
                    self._perturb(population[k], initial_result)

                if self._search_over:
                    break

            # Check if there is any successful attack in the current population
            top_member = max(population, key=lambda x: x.score)
            if (
                self._search_over
                or top_member.result.goal_status == GoalFunctionResultStatus.SUCCEEDED
            ):
                return top_member.result

            # Update the elite if the score is increased
            for k in range(len(population)):
                if population[k].score > local_elites[k].score:
                    local_elites[k] = copy.copy(population[k])

            if top_member.score > global_elite.score:
                global_elite = copy.copy(top_member)

        return global_elite.result

    def check_transformation_compatibility(self, transformation):
        """The genetic algorithm is specifically designed for word
        substitutions."""
        return transformation_consists_of_word_swaps(transformation)

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["pop_size", "max_iters", "post_turn_check", "max_turn_retries"]


def normalize(n):
    n = np.array(n)
    n[n < 0] = 0
    s = np.sum(n)
    if s == 0:
        return np.ones(len(n)) / len(n)
    else:
        return n / s
