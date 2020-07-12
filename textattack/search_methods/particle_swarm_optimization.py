"""
Reimplementation of search method from Word-level Textual Adversarial Attacking as Combinatorial Optimization
by Zang et. al
`<https://www.aclweb.org/anthology/2020.acl-main.540.pdf>`_
`<https://github.com/thunlp/SememePSO-Attack>`_
"""

from copy import deepcopy

import numpy as np

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod


class ParticleSwarmOptimization(SearchMethod):
    """
    Attacks a model with word substiutitions using a Particle Swarm Optimization (PSO) algorithm.
    Some key hyper-parameters are setup according to the original paper:

    "We adjust PSO on the validation set of SST and set ω_1 as 0.8 and ω_2 as 0.2.
    We set the max velocity of the particles V_{max} to 3, which means the changing
    probability of the particles ranges from 0.047 (sigmoid(-3)) to 0.953 (sigmoid(3))."

    Args:
        pop_size (:obj:`int`, optional): The population size. Defauls to 60.
        max_iters (:obj:`int`, optional): The maximum number of iterations to use. Defaults to 20.
        post_movement_check (bool): If True, check if new position reached by moving passes the constraints.
        max_movement_retries (int): Maximum number of movement retries if new position fails to pass the constraints.
            Applied only when `post_movement_check` is set to `True`.
            Setting it to 0 means we immediately take the old position as the new position upon failure.
    """

    def __init__(
        self, pop_size=60, max_iters=20, post_movement_check=True
    ):
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.post_movement_check = post_movement_check

        self._search_over = False
        self.omega_1 = 0.8
        self.omega_2 = 0.2
        self.C1_origin = 0.8
        self.C2_origin = 0.2
        self.V_max = 3.0

    def _turn(self, x1, x2, prob, x_len):
        indices_to_replace = []
        words_to_replace = []
        x2_words = x2.words
        for i in range(x_len):
            if np.random.uniform() < prob[i]:
                indices_to_replace.append(i)
                words_to_replace.append(x2_words[i])
        new_text = x1.replace_words_at_indices(indices_to_replace, words_to_replace)
        return new_text

    def _norm(self, n):
        n = [max(0, i) for i in n]
        s = sum(n)
        if s == 0:
            return [1 / len(n) for _ in n]
        else:
            return [i / s for i in n]

    def _equal(self, a, b):
        return -self.V_max if a == b else self.V_max

    def _count_change_ratio(self, x1, x2, x_len):
        return float(np.sum(x1.words != x2.words)) / float(x_len)

    def _sigmoid(self, n):
        return 1 / (1 + np.exp(-n))

    def _gen_most_change(self, x_cur, pos, replace_list):
        orig_result, self._search_over = self.get_goal_results([x_cur])
        if self._search_over:
            return 0, x_cur.words[pos]
        new_x_list = [x_cur.replace_word_at_index(pos, w) for w in replace_list]
        # new_x_list = self.get_transformations(
        #     x_cur,
        #     original_text=self.original_attacked_text,
        #     indices_to_modify=[pos],
        # )
        new_x_results, self._search_over = self.get_goal_results(new_x_list)
        new_x_scores = np.array([r.score for r in new_x_results])
        new_x_scores = (
            new_x_scores - orig_result[0].score
        )  # minimize the score of ground truth
        if len(new_x_scores):
            return (
                np.max(new_x_scores),
                new_x_list[np.argsort(new_x_scores)[-1]].words[pos],
            )
        else:
            return 0, x_cur.words[pos]

    def _gen_h_score(self, current_text, neighbors_list):
        current_result, self._search_over = self.get_goal_results([current_text])
        current_result = current_result[0]
        if self._search_over:
            return None, None
        candidate_list = []
        score_list = []
        for i in range(len(neighbors_list)):
            if not neighbors_list[i]:
                candidate_list.append(current_result)
                score_list.append(0)
                continue

            neighbor_results, self._search_over = self.get_goal_results(neighbors_list)
            if self._search_over:

            neighbor_scores = np.array([r.score for in neighbor_results])
            score_diff = neighbor_scores - current_result.score
            best_neighbor = np.argmax(neighbor_scores)[0]
            candidate_list.append(neighbors_list[best_neighbor])
            score_list.append(score_diff[best_neighbor])

        prob_list = self._norm(score_list)

        return candidate_list, prob_list

    def _get_neighbors_list(self, original_text):
        """
        Generates this neighbors_len list
        Args:
            original_text (AttackedText): The original text
        Returns:
            `list[list[AttackedText]]` representing list of candidate neighbors for each word
        """
        words = attacked_text.words
        neighbors_list = [[] for _ in range(len(words))]
        transformations = self.get_transformations(
            original_text, original_text=original_text
        )
        for transformed_text in transformations:
            try:
                diff_idx = next(iter(transformed_text.attack_attrs["newly_modified_indices"]))
                neighbors_list[diff_idx].append(transformed_text)
            except:
                assert len(attacked_text.words) == len(transformed_text.words)
                assert all(
                    [
                        w1 == w2
                        for w1, w2 in zip(attacked_text.words, transformed_text.words)
                    ]
                )
        return neighbors_list

    def _mutate(self, current_text, original_text):
        neighbors_list = self._get_neighbors_list(current_text)
        candidate_list, prob_list = self._gen_h_score(neighbors_list, original_text)
        random_candidate = np.random.choice(candidate_list, 1, p=prob_list)[0]
        return random_candidate

    def _generate_population(self, initial_result):
        neighbors_list = self._get_neighbors_list(initial_result.attacked_text)
        candidate_list, prob_list = self._gen_h_score(neighbors_list, original_text)
        population = []
        for _ in range(self.pop_size):
            # Mutation step
            random_candidate = np.random.choice(candidate_list, 1, p=prob_list)[0]
            population.append(random_candidate)

        return population

    def _perform_search(self, initial_result):
        x_len = len(initial_result.attacked_text.words)
        # get word substitute candidates and generate population
        neighbors_list = self._get_neighbors_list(initial_result.attacked_text)
        neighbors_len = [len(x) for x in neighbors_list]
        population = self._generate_population(initial_result)

        # test population against target model
        pop_results, self._search_over = self.get_goal_results(pop)
        if self._search_over:
            return max(pop_results, key=lambda x: x.score)
        pop_scores = np.array([r.score for r in pop_results])

        # rank the scores from low to high and check if there is a successful attack
        part_elites = deepcopy(pop)
        part_elites_scores = pop_scores
        top_attack = np.argmax(pop_scores)
        all_elite = pop[top_attack]
        all_elite_score = pop_scores[top_attack]
        if pop_results[top_attack].goal_status == GoalFunctionResultStatus.SUCCEEDED:
            return pop_results[top_attack]

        # set up hyper-parameters
        V = np.random.uniform(-self.V_max, self.V_max, self.pop_size)
        V_P = [[V[t] for _ in range(x_len)] for t in range(self.pop_size)]

        # start iterations
        for i in range(self.max_iters):

            omega = (self.omega_1 - self.omega_2) * (
                self.max_iters - i
            ) / self.max_iters + self.omega_2
            C1 = self.C1_origin - i / self.max_iters * (self.C1_origin - self.C2_origin)
            C2 = self.C2_origin + i / self.max_iters * (self.C1_origin - self.C2_origin)
            P1 = C1
            P2 = C2

            all_elite_words = all_elite.words
            for id in range(self.pop_size):

                # calculate the probability of turning each word
                pop_words = pop[id].words
                part_elites_words = part_elites[id].words
                for dim in range(x_len):
                    V_P[id][dim] = omega * V_P[id][dim] + (1 - omega) * (
                        self._equal(pop_words[dim], part_elites_words[dim])
                        + self._equal(pop_words[dim], all_elite_words[dim])
                    )
                turn_prob = [self._sigmoid(V_P[id][d]) for d in range(x_len)]

                if np.random.uniform() < P1:
                    pop[id] = self._turn(part_elites[id], pop[id], turn_prob, x_len)
                if np.random.uniform() < P2:
                    pop[id] = self._turn(all_elite, pop[id], turn_prob, x_len)

            # check if there is any successful attack in the current population
            pop_results, self._search_over = self.get_goal_results(pop)
            if self._search_over:
                return max(pop_results, key=lambda x: x.score)
            pop_scores = np.array([r.score for r in pop_results])
            top_attack = np.argmax(pop_scores)
            if (
                pop_results[top_attack].goal_status
                == GoalFunctionResultStatus.SUCCEEDED
            ):
                return pop_results[top_attack]

            # mutation based on the current change rate
            new_pop = []
            for x in pop:
                change_ratio = self._count_change_ratio(
                    x, self.original_attacked_text, x_len
                )
                p_change = (
                    1 - 2 * change_ratio
                )  # referred from the original source code
                if np.random.uniform() < p_change:
                    new_h, new_w_list = self._gen_h_score(
                        x, neighbors_len, neighbors_list
                    )
                    new_pop.append(self._mutate(x, new_h, new_w_list))
                else:
                    new_pop.append(x)
            pop = new_pop

            # check if there is any successful attack in the current population
            pop_results, self._search_over = self.get_goal_results(pop)
            if self._search_over:
                return max(pop_results, key=lambda x: x.score)
            pop_scores = np.array([r.score for r in pop_results])
            top_attack = np.argmax(pop_scores)
            if (
                pop_results[top_attack].goal_status
                == GoalFunctionResultStatus.SUCCEEDED
            ):
                return pop_results[top_attack]

            # update the elite if the score is increased
            for k in range(self.pop_size):
                if pop_scores[k] > part_elites_scores[k]:
                    part_elites[k] = pop[k]
                    part_elites_scores[k] = pop_scores[k]
            if pop_scores[top_attack] > all_elite_score:
                all_elite = pop[top_attack]
                all_elite_score = pop_scores[top_attack]

        return initial_result

class Particle:
    def __init__(attacked_text):
        self.attacked_text