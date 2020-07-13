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


class ParticleSwarmOptimization(SearchMethod):
    """Attacks a model with word substiutitions using a Particle Swarm
    Optimization (PSO) algorithm. Some key hyper-parameters are setup according
    to the original paper:

    "We adjust PSO on the validation set of SST and set ω_1 as 0.8 and ω_2 as 0.2.
    We set the max velocity of the particles V_{max} to 3, which means the changing
    probability of the particles ranges from 0.047 (sigmoid(-3)) to 0.953 (sigmoid(3))."

    Args:
        pop_size (:obj:`int`, optional): The population size. Defauls to 60.
        max_iters (:obj:`int`, optional): The maximum number of iterations to use. Defaults to 20.
    """

    def __init__(
        self, pop_size=60, max_iters=20,
    ):
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.search_over = False

        self.Omega_1 = 0.8
        self.Omega_2 = 0.2
        self.C1_origin = 0.8
        self.C2_origin = 0.2
        self.V_max = 3.0

    def _generate_population(self, x_orig, neighbors_list, neighbors_len):
        h_score, w_list = self._gen_h_score(x_orig, neighbors_len, neighbors_list)
        return [self._mutate(x_orig, h_score, w_list) for _ in range(self.pop_size)]

    def _mutate(self, x_cur, w_select_probs, w_list):
        rand_idx = np.random.choice(len(w_select_probs), 1, p=w_select_probs)[0]
        return x_cur.replace_word_at_index(rand_idx, w_list[rand_idx])

    def _gen_h_score(self, x, neighbors_len, neighbors_list):

        w_list = []
        prob_list = []
        for i, orig_w in enumerate(x.words):
            if neighbors_len[i] == 0:
                w_list.append(orig_w)
                prob_list.append(0)
                continue
            p, w = self._gen_most_change(x, i, neighbors_list[i])
            w_list.append(w)
            prob_list.append(p)

        prob_list = self._norm(prob_list)

        h_score = prob_list
        h_score = np.array(h_score)
        return h_score, w_list

    def _norm(self, n):

        tn = []
        for i in n:
            if i <= 0:
                tn.append(0)
            else:
                tn.append(i)
        s = np.sum(tn)
        if s == 0:
            for i in range(len(tn)):
                tn[i] = 1
            return [t / len(tn) for t in tn]
        new_n = [t / s for t in tn]

        return new_n

    # for un-targeted attacking
    def _gen_most_change(self, x_cur, pos, replace_list):
        orig_result, self.search_over = self.get_goal_results([x_cur])
        if self.search_over:
            return 0, x_cur.words[pos]
        new_x_list = [x_cur.replace_word_at_index(pos, w) for w in replace_list]
        # new_x_list = self.get_transformations(
        #     x_cur,
        #     original_text=self.original_attacked_text,
        #     indices_to_modify=[pos],
        # )
        new_x_results, self.search_over = self.get_goal_results(new_x_list)
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

    def _get_neighbors_list(self, attacked_text):
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
            try:
                diff_idx = attacked_text.first_word_diff_index(transformed_text)
                neighbors_list[diff_idx].append(transformed_text.words[diff_idx])
            except Exception:
                assert len(attacked_text.words) == len(transformed_text.words)
                assert all(
                    [
                        w1 == w2
                        for w1, w2 in zip(attacked_text.words, transformed_text.words)
                    ]
                )
        neighbors_list = [np.array(x) for x in neighbors_list]
        return neighbors_list

    def _equal(self, a, b):
        if a == b:
            return -self.V_max
        else:
            return self.V_max

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

    def _count_change_ratio(self, x1, x2, x_len):
        change_ratio = float(np.sum(x1.words != x2.words)) / float(x_len)
        return change_ratio

    def _sigmoid(self, n):
        return 1 / (1 + np.exp(-n))

    def _perform_search(self, initial_result):
        self.original_attacked_text = initial_result.attacked_text
        x_len = len(self.original_attacked_text.words)
        self.correct_output = initial_result.output

        # get word substitute candidates and generate population
        neighbors_list = self._get_neighbors_list(self.original_attacked_text)
        neighbors_len = [len(x) for x in neighbors_list]
        pop = self._generate_population(
            self.original_attacked_text, neighbors_list, neighbors_len
        )

        # test population against target model
        pop_results, self.search_over = self.get_goal_results(pop)
        if self.search_over:
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

            Omega = (self.Omega_1 - self.Omega_2) * (
                self.max_iters - i
            ) / self.max_iters + self.Omega_2
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
                    V_P[id][dim] = Omega * V_P[id][dim] + (1 - Omega) * (
                        self._equal(pop_words[dim], part_elites_words[dim])
                        + self._equal(pop_words[dim], all_elite_words[dim])
                    )
                turn_prob = [self._sigmoid(V_P[id][d]) for d in range(x_len)]

                if np.random.uniform() < P1:
                    pop[id] = self._turn(part_elites[id], pop[id], turn_prob, x_len)
                if np.random.uniform() < P2:
                    pop[id] = self._turn(all_elite, pop[id], turn_prob, x_len)

            # check if there is any successful attack in the current population
            pop_results, self.search_over = self.get_goal_results(pop)
            if self.search_over:
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
            pop_results, self.search_over = self.get_goal_results(pop)
            if self.search_over:
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
