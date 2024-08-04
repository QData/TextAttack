"""

LEAP Particle Swarm Optimization
====================================

LEAP, an automated test method that uses LEvy flight-based Adaptive Particle
swarm optimization integrated with textual features to generate adversarial test cases.

al
`<https://arxiv.org/abs/2308.11284>`_
`<https://github.com/lumos-xiao/LEAP>`_
"""

import copy

import numpy as np
from scipy.special import gamma as gamma

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import ParticleSwarmOptimization


def sigmax(alpha):
    numerator = gamma(alpha + 1.0) * np.sin(np.pi * alpha / 2.0)
    denominator = gamma((alpha + 1) / 2.0) * alpha * np.power(2.0, (alpha - 1.0) / 2.0)
    return np.power(numerator / denominator, 1.0 / alpha)


def vf(alpha):
    x = np.random.normal(0, 1)
    y = np.random.normal(0, 1)

    x = x * sigmax(alpha)

    return x / np.power(np.abs(y), 1.0 / alpha)


def K(alpha):
    k = alpha * gamma((alpha + 1.0) / (2.0 * alpha)) / gamma(1.0 / alpha)
    k *= np.power(
        alpha
        * gamma((alpha + 1.0) / 2.0)
        / (gamma(alpha + 1.0) * np.sin(np.pi * alpha / 2.0)),
        1.0 / alpha,
    )

    return k


def C(alpha):
    x = np.array(
        (0.75, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.95, 1.99)
    )
    y = np.array(
        (
            2.2085,
            2.483,
            2.7675,
            2.945,
            2.941,
            2.9005,
            2.8315,
            2.737,
            2.6125,
            2.4465,
            2.206,
            1.7915,
            1.3925,
            0.6089,
        )
    )

    return np.interp(alpha, x, y)


def levy(alpha, gamma=1, n=1):
    w = 0
    for i in range(0, n):
        v = vf(alpha)

        while v < -10:
            v = vf(alpha)

        w += v * ((K(alpha) - 1.0) * np.exp(-v / C(alpha)) + 1.0)

    z = 1.0 / np.power(n, 1.0 / alpha) * w * gamma

    return z


def get_one_levy(min, max):
    while True:
        temp = levy(1.5, 1)
        if min <= temp <= max:
            break
        else:
            continue
    return temp


def softmax(x, axis=1):
    row_max = x.max(axis=axis)

    # Each element of the row needs to be subtracted from the corresponding maximum value, otherwise exp(x) will overflow, resulting in the inf case
    row_max = row_max.reshape(-1, 1)
    x = x - row_max

    # Calculate the exponential power of e
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


class ParticleSwarmOptimizationLEAP(ParticleSwarmOptimization):
    """Attacks a model with word substiutitions using a variant of Particle
    Swarm Optimization (PSO) algorithm called LEAP."""

    def _greedy_perturb(self, pop_member, original_result):
        best_neighbors, prob_list = self._get_best_neighbors(
            pop_member.result, original_result
        )
        random_result = best_neighbors[np.argsort(prob_list)[-1]]
        pop_member.attacked_text = random_result.attacked_text
        pop_member.result = random_result
        return True

    def perform_search(self, initial_result):
        self._search_over = False
        population = self._initialize_population(initial_result, self.pop_size)

        # Initialize velocities
        v_init = []
        v_init_rand = np.random.uniform(-self.v_max, self.v_max, self.pop_size)
        v_init_levy = []
        while True:
            temp = levy(1.5, 1)
            if -self.v_max <= temp <= self.v_max:
                v_init_levy.append(temp)
            else:
                continue
            if len(v_init_levy) == self.pop_size:
                break
        for i in range(self.pop_size):
            if np.random.uniform(
                -self.v_max,
                self.v_max,
            ) < levy(1.5, 1):
                v_init.append(v_init_rand[i])
            else:
                v_init.append(v_init_levy[i])
        v_init = np.array(v_init)

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

        pop_fit_list = []
        for i in range(len(population)):
            pop_fit_list.append(population[i].score)
        pop_fit = np.array(pop_fit_list)
        fit_ave = round(pop_fit.mean(), 3)
        fit_min = pop_fit.min()

        # start iterations
        omega = []
        for i in range(self.max_iters):
            for k in range(len(population)):
                if population[k].score < fit_ave:
                    omega.append(
                        self.omega_2
                        + (
                            (population[k].score - fit_min)
                            * (self.omega_1 - self.omega_2)
                        )
                        / (fit_ave - fit_min)
                    )
                else:
                    omega.append(get_one_levy(0.5, 0.8))
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
                    velocities[k][d] = omega[k] * velocities[k][d] + (1 - omega[k]) * (
                        self._equal(pop_mem_words[d], local_elite_words[d])
                        + self._equal(pop_mem_words[d], global_elite.words[d])
                    )
                turn_list = np.array([velocities[k]])
                turn_prob = softmax(turn_list)[0]

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
