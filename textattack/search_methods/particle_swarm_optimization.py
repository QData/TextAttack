"""Reimplementation of search method from Word-level Textual Adversarial
Attacking as Combinatorial Optimization by Zang et.

al
`<https://www.aclweb.org/anthology/2020.acl-main.540.pdf>`_
`<https://github.com/thunlp/SememePSO-Attack>`_
"""

from copy import deepcopy

import numpy as np

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import transformation_consists_of_word_swaps


class ParticleSwarmOptimization(SearchMethod):
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
        self.C1_origin = 0.8
        self.C2_origin = 0.2
        self.V_max = 3.0

    def _norm(self, n):
        n = [max(0, i) for i in n]
        s = sum(n)
        if s == 0:
            return [1 / len(n) for _ in n]
        else:
            return [i / s for i in n]

    def _equal(self, a, b):
        return -self.V_max if a == b else self.V_max

    def _count_change_ratio(self, x1, x2):
        return float(np.sum(x1.words != x2.words)) / float(len(x2.words))

    def _sigmoid(self, n):
        return 1 / (1 + np.exp(-n))

    def _turn(self, source_x, target_x, prob, original_text):
        """
        Based on given probability distribution, "move" to `target_x` from `source_x`
        Args:
            source_x (Position): Text we start from.
            target_x (Position): Text we want to move to.
            prob (list[float]): Turn probability for each word.
            original_text (AttackedText): Original text for constraint check if `self.post_turn_check=True`.
        Returns:
            New `Position` that we moved to (or if we fail to move, same as `source_x`)
        """
        assert len(source_x.words) == len(
            target_x.words
        ), "Word length mismatch for turn operation."
        assert len(source_x.words) == len(
            prob
        ), "Length mistmatch for words and probability list."
        len_x = len(source_x.words)

        num_tries = 0
        passed_constraints = False
        while num_tries < self.max_turn_retries + 1:
            indices_to_replace = []
            words_to_replace = []
            for i in range(len_x):
                if np.random.uniform() < prob[i]:
                    indices_to_replace.append(i)
                    words_to_replace.append(target_x.words[i])
            new_text = source_x.attacked_text.replace_words_at_indices(
                indices_to_replace, words_to_replace
            )

            if not self.post_turn_check or (new_text.words == source_x.words):
                break

            if "last_transformation" in source_x.attacked_text.attack_attrs:
                new_text.attack_attrs[
                    "last_transformation"
                ] = source_x.attacked_text.attack_attrs["last_transformation"]
                filtered = self.filter_transformations(
                    [new_text], source_x.attacked_text, original_text=original_text
                )
            else:
                # In this case, source_x has not been transformed,
                # meaning that new_text = source_x = original_text
                filtered = [new_text]

            if filtered:
                new_text = filtered[0]
                passed_constraints = True
                break

            num_tries += 1

        if self.post_turn_check and not passed_constraints:
            # If we cannot find a turn that passes the constraints, we do not move.
            return source_x
        else:
            return Position(new_text)

    def _get_best_neighbors(self, neighbors_list, current_position):
        """
        For given `current_position`, find the neighboring position that yields
        maximum improvement (in goal function score) for each word.
        Args:
            neighbors_list (list[list[AttackedText]]): List of "neighboring" AttackedText for each word in `current_text`.
            current_position (Position): Current position
        Returns:
            best_neighbors (list[Position]): Best neighboring positions for each word
            prob_list (list[float]): discrete probablity distribution for sampling a neighbor from `best_neighbors`
        """
        best_neighbors = []
        score_list = []
        for i in range(len(neighbors_list)):
            if not neighbors_list[i]:
                best_neighbors.append(current_position)
                score_list.append(0)
                continue

            neighbor_results, self._search_over = self.get_goal_results(
                neighbors_list[i]
            )
            if neighbor_results:
                # This is incase query budget forces len(neighbor_results) == 0
                neighbor_scores = np.array([r.score for r in neighbor_results])
                score_diff = neighbor_scores - current_position.score
                best_idx = np.argmax(neighbor_scores)
                best_neighbors.append(
                    Position(neighbors_list[i][best_idx], neighbor_results[best_idx])
                )
                score_list.append(score_diff[best_idx])

            if self._search_over:
                break

        prob_list = self._norm(score_list)

        return best_neighbors, prob_list

    def _get_neighbors_list(self, current_text, original_text):
        """
        For each word in `current_text`, find list of available transformations.
        Args:
            current_text (AttackedText): Current text
            original_text (AttackedText): Original text for constraint check
        Returns:
            `list[list[AttackedText]]` representing list of candidate neighbors for each word
        """
        neighbors_list = [[] for _ in range(len(current_text.words))]
        transformations = self.get_transformations(
            current_text, original_text=original_text
        )
        for transformed_text in transformations:
            diff_idx = next(
                iter(transformed_text.attack_attrs["newly_modified_indices"])
            )
            neighbors_list[diff_idx].append(transformed_text)

        return neighbors_list

    def _mutate(self, current_position, original_text):
        neighbors_list = self._get_neighbors_list(
            current_position.attacked_text, original_text
        )
        candidate_list, prob_list = self._get_best_neighbors(
            neighbors_list, current_position
        )
        if self._search_over:
            return current_position
        random_candidate = np.random.choice(candidate_list, 1, p=prob_list)[0]
        return random_candidate

    def _generate_population(self, initial_position):
        """
        Generate population of particles (represented by `Position`) from `initial_position`
        Args:
            initial_position (Position): Original text represented as `Position`
        Returns:
            `list[Position]` representing population
        """
        neighbors_list = self._get_neighbors_list(
            initial_position.attacked_text, initial_position.attacked_text
        )
        best_neighbors, prob_list = self._get_best_neighbors(
            neighbors_list, initial_position
        )
        population = []
        for _ in range(self.pop_size):
            # Mutation step
            random_position = np.random.choice(best_neighbors, 1, p=prob_list)[0]
            population.append(random_position)
        return population

    def _perform_search(self, initial_result):
        self._search_over = False
        original_position = Position(initial_result.attacked_text, initial_result)
        # get word substitute candidates and generate population
        population = self._generate_population(original_position)
        global_elite = max(population, key=lambda x: x.score)
        if (
            self._search_over
            or global_elite.result.goal_status == GoalFunctionResultStatus.SUCCEEDED
        ):
            return global_elite.result

        # rank the scores from low to high and check if there is a successful attack
        local_elites = deepcopy(population)
        # set up hyper-parameters
        V = np.random.uniform(-self.V_max, self.V_max, self.pop_size)
        V_P = [
            [V[t] for _ in range(len(initial_result.attacked_text.words))]
            for t in range(self.pop_size)
        ]

        # start iterations
        for i in range(self.max_iters):
            omega = (self.omega_1 - self.omega_2) * (
                self.max_iters - i
            ) / self.max_iters + self.omega_2
            C1 = self.C1_origin - i / self.max_iters * (self.C1_origin - self.C2_origin)
            C2 = self.C2_origin + i / self.max_iters * (self.C1_origin - self.C2_origin)
            P1 = C1
            P2 = C2

            for k in range(len(population)):
                # calculate the probability of turning each word
                particle_words = population[k].words
                local_elite_words = local_elites[k].words
                assert len(particle_words) == len(
                    local_elite_words
                ), "PSO word length mismatch!"
                for dim in range(len(particle_words)):
                    V_P[k][dim] = omega * V_P[k][dim] + (1 - omega) * (
                        self._equal(particle_words[dim], local_elite_words[dim])
                        + self._equal(particle_words[dim], local_elite_words[dim])
                    )
                turn_prob = [
                    self._sigmoid(V_P[k][d]) for d in range(len(particle_words))
                ]

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
            for k in range(len(population)):
                population[k].result = pop_results[k]
            top_result = max(population, key=lambda x: x.score).result
            if (
                self._search_over
                or top_result.goal_status == GoalFunctionResultStatus.SUCCEEDED
            ):
                return top_result

            # Mutation based on the current change rate
            for k in range(len(population)):
                p = population[k]
                change_ratio = self._count_change_ratio(p, initial_result.attacked_text)
                # Referred from the original source code
                p_change = 1 - 2 * change_ratio
                if np.random.uniform() < p_change:
                    population[k] = self._mutate(p, initial_result.attacked_text)

                if self._search_over:
                    break
            if self._search_over:
                return top_result

            # Check if there is any successful attack in the current population
            pop_results, self._search_over = self.get_goal_results(
                [p.attacked_text for p in population]
            )
            for k in range(len(population)):
                population[k].result = pop_results[k]
            top_result = max(population, key=lambda x: x.score).result
            if (
                self._search_over
                or top_result.goal_status == GoalFunctionResultStatus.SUCCEEDED
            ):
                return top_result

            # Update the elite if the score is increased
            for k in range(len(population)):
                if population[k].score > local_elites[k].score:
                    local_elites[k] = deepcopy(population[k])

            if top_result.score > global_elite.score:
                elite_result = deepcopy(top_result)
                global_elite = Position(elite_result.attacked_text, elite_result)

        return global_elite.result

    def check_transformation_compatibility(self, transformation):
        """The genetic algorithm is specifically designed for word
        substitutions."""
        return transformation_consists_of_word_swaps(transformation)

    def extra_repr_keys(self):
        return ["pop_size", "max_iters", "post_turn_check", "max_turn_retries"]


class Position:
    """
    Helper class for particle-swarm optimization.
    Each position represents transformed version of original text
    Args:
        attacked_text (:obj:`AttackedText`): `AttackedText` for the transformed text
        result (:obj:`GoalFunctionResult`, optional): `GoalFunctionResult` for the transformed text
    """

    def __init__(self, attacked_text, result=None):
        self.attacked_text = attacked_text
        self.result = result

    @property
    def score(self):
        if not self.result:
            raise ValueError('"result" attribute undefined for Position')
        return self.result.score

    @property
    def words(self):
        return self.attacked_text.words
