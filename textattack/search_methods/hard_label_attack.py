"""Adversarial Attacks in a Hard Label Black Box Setting
====================================

Reimplementation of search method from Generating Natural Language Attack in a Hard Label Black Box Setting by Maheshwary et.al

`https://arxiv.org/abs/2012.14956`_

`<https://github.com/RishabhMaheshwary/hard-label-attack>`_
"""

from collections import defaultdict

import numpy as np
from scipy.special import softmax
import torch

from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import PopulationBasedSearch, PopulationMember
from textattack.shared.validators import transformation_consists_of_word_swaps


class HardLabelAttack(PopulationBasedSearch):

    """Attacks a model with word substitutions by observing only the topmost
    predicted label. It consist of three key steps: (1) Random Initialization
    (2) Search Space Reduction and (3) Genetic Based Optimization. We use the
    hyper-parameters values as mentioned in the original paper. Following the
    original implementation of the paper we set the window size used for
    measuring semantic similarity as 40 and the trials used for generating a
    random adversarial example to 2500. The trials can be reduced to speed up
    attack but it will lower down the  success rate of the attack.

    Args:
        pop_size (:obj:`int`, optional): The population size. Defaults to 30.
        max_iters (:obj:`int`, optional): The maximum number of iterations to use. Defaults to 100.
        max_replacements_per_index (:obj:`int`, optional): Maximum number of mutations that can be sone on an index. Defaults to 25.
        window_size (:obj:`int`, optional): The window of text around the changed index to be considered for measuring semantic similarity. Defaults to 40.
        trials (:obj:`int`, optional): The number of trials for generating random adversarial example. Defaults to 2500.
    """

    def __init__(
        self,
        pop_size=30,
        max_iters=100,
        max_replacements_per_index=25,
        window_size=40,
        trials=2500,
    ):
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.trials = trials
        self.max_replacements_per_index = max_replacements_per_index

        self._search_over = False
        self.window_size = window_size
        self.sim_metric = torch.nn.CosineSimilarity(dim=1)
        self.use = UniversalSentenceEncoder()

    def _get_similarity_score(
        self, transformed_texts, initial_text, window_around_idx=-1, use_window=False
    ):
        """
        Calculates semantic similarity betweem the `transformed_texts` and `initial_text`
        Args:
            transformed_texts (list): The perturbed or mutated text inputs.
            initial_text (AttackedText): The original text input.
            window_around_idx (int): The index around which the window to consider.
            use_window (bool): Calculate similarity with or without window.
        Returns:
            scores as `list[float]`
        """

        if use_window:
            texts_within_window = []
            for txt in transformed_texts:
                texts_within_window.append(
                    txt.text_window_around_index(window_around_idx, self.window_size)
                )
            transformed_texts = texts_within_window
            initial_text = initial_text.text_window_around_index(
                window_around_idx, self.window_size
            )

        embeddings_transformed_texts = self.use.encode(transformed_texts)
        embeddings_initial_text = self.use.encode(
            [initial_text] * len(transformed_texts)
        )

        if not isinstance(embeddings_transformed_texts, torch.Tensor):
            embeddings_transformed_texts = torch.tensor(embeddings_transformed_texts)

        if not isinstance(embeddings_initial_text, torch.Tensor):
            embeddings_initial_text = torch.tensor(embeddings_initial_text)

        scores = self.sim_metric(embeddings_transformed_texts, embeddings_initial_text)
        return scores

    def _perturb(
        self, pop_member, original_result, initial_text, index_to_perturb, best_attack
    ):
        """
        Mutates or perturbes a population member `pop_member` at `index_to_perturb`
        Args:
            pop_member (PopulationMember): The pop_member to perturb or mutate.
            original_result (GoalFunctionResult): The result of `pop_member`.
            initial_text (AttackedText): The original text input.
            index_to_perturb (int): The index to be perturbed or mutated.
            best_attack (PopulationMember): The best attack until now.
        Returns:
            pop_member as `PopulationMember`
        """

        cur_adv_text = pop_member.attacked_text
        # First try to replace the substituted word back with the original word.
        perturbed_text = cur_adv_text.replace_word_at_index(
            index_to_perturb, initial_text.words[index_to_perturb]
        )
        results, search_over = self.get_goal_results([perturbed_text])
        if not search_over:
            return PopulationMember(
                cur_adv_text,
                attributes={
                    "similarity_score": -1,
                    "valid_adversarial_example": False,
                },
            )
        if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
            similarity_score = self._get_similarity_score(
                [perturbed_text.text], initial_text.text
            )[0]
            return PopulationMember(
                perturbed_text,
                attributes={
                    "similarity_score": similarity_score,
                    "valid_adversarial_example": True,
                },
            )
        # Replace with other synonyms.
        perturbed_texts = self.get_transformations(
            perturbed_text,
            original_text=initial_text,
            indices_to_modify=[index_to_perturb],
        )

        final_perturbed_texts = []

        for txt in perturbed_texts:

            passed_constraints = self._check_constraints(
                txt, perturbed_text, original_text=initial_text
            )

            if (
                passed_constraints
                and txt.words[index_to_perturb] != cur_adv_text.words[index_to_perturb]
            ):
                final_perturbed_texts.append(txt)

        results, _ = self.get_goal_results(final_perturbed_texts)

        perturbed_adv_texts = []
        perturbed_adv_strings = []
        for i in range(len(results)):

            if results[i].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                perturbed_adv_texts.append(final_perturbed_texts[i])
                perturbed_adv_strings.append(final_perturbed_texts[i].text)

        if len(perturbed_adv_texts) == 0:
            return PopulationMember(
                cur_adv_text,
                attributes={
                    "similarity_score": pop_member.attributes["similarity_score"],
                    "valid_adversarial_example": False,
                },
            )

        prev_similarity_scores = self._get_similarity_score(
            [pop_member.attacked_text], initial_text, index_to_perturb, use_window=True
        )
        similarity_scores = self._get_similarity_score(
            perturbed_adv_texts, initial_text, index_to_perturb, use_window=True
        )

        best_similarity_score = -1
        for i in range(len(similarity_scores)):
            # Filter out candidates which do not improve the semantic similarity
            # score within `self.window_size`.
            if similarity_scores[i] < prev_similarity_scores[0]:
                continue

            if similarity_scores[i] > best_similarity_score:
                best_similarity_score = similarity_scores[i]
                final_adv_text = perturbed_adv_texts[i]

        if best_similarity_score == -1:
            return PopulationMember(
                cur_adv_text,
                attributes={
                    "similarity_score": best_similarity_score,
                    "valid_adversarial_example": False,
                },
            )

        new_pop_member = PopulationMember(
            final_adv_text,
            attributes={
                "similarity_score": best_similarity_score,
                "valid_adversarial_example": True,
            },
        )
        return new_pop_member

    def _initialize_population(self, cur_adv_text, initial_text, changed_indices):
        """
        Initialize a population of size `pop_size` by mutating the indices in `changed_indices`
        Args:
            cur_adv_text (AttackedText): The adversarial text obtained after search space reduction step.
            initial_text (AttackedText): The original input text.
            changed_indices (list): The indices changed in the `cur_adv_text` from the `initial_text`.
        Returns:
            population as `list[PopulationMember]`
        """
        results, _ = self.get_goal_results([cur_adv_text])
        similarity_score = self._get_similarity_score(
            [cur_adv_text.text], initial_text.text
        )[0]

        initial_pop_member = PopulationMember(
            cur_adv_text,
            attributes={
                "similarity_score": similarity_score,
                "valid_adversarial_example": True,
            },
        )
        population = []
        # Randomly select `self.pop_size - 1` substituted indices for mutation.
        indices = np.random.choice(len(changed_indices), size=self.pop_size - 1)
        for i in range(len(indices)):

            new_pop_member = self._perturb(
                initial_pop_member,
                results[0],
                initial_text,
                changed_indices[indices[i]],
                initial_pop_member,
            )

            if new_pop_member.attributes["valid_adversarial_example"]:
                population.append(new_pop_member)

        return population

    def _generate_initial_adversarial_example(self, initial_result):
        """Generates a random initial adversarial example.

        Args:
            initial_result (GoalFunctionResult): Original result.
        Returns:
            initial_adversarial_text as `AttackedText`
        """
        initial_text = initial_result.attacked_text
        len_text = len(initial_text.words)

        for trial in range(self.trials):

            random_index_order = np.arange(len_text)
            np.random.shuffle(random_index_order)
            random_text = initial_text
            random_generated_texts = []
            for idx in random_index_order:

                transformed_texts = self.get_transformations(
                    random_text, original_text=initial_text, indices_to_modify=[idx]
                )

                if len(transformed_texts) > 0:
                    random_text = np.random.choice(transformed_texts)
                    random_generated_texts.append(random_text)
            results, _ = self.get_goal_results(random_generated_texts)

            for i in range(len(results)):
                if results[i].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    return results[i].attacked_text, results[i]
        return initial_text, initial_result

    def _search_space_reduction(self, initial_result, initial_adv_text):
        """Reduces the count of perturbed words in the `initial_adv_text`.

        Args:
            initial_result (GoalFunctionResult): Original result.
            initial_adv_text (int): The adversarial text obtained after random initialization.
        Returns:
            initial_adv_text as `AttackedText`
        """
        original_text = initial_result.attacked_text
        different_indices = original_text.all_words_diff(initial_adv_text)
        replacements = []
        txts = []
        # Replace the substituted words with their original counterparts.
        for idx in different_indices:

            new_text = initial_adv_text.replace_word_at_index(
                idx, initial_result.attacked_text.words[idx]
            )
            # Filter out replacements that are not adversarial.
            results, search_over = self.get_goal_results([new_text])
            if not search_over:
                return initial_adv_text
            if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                txts.append(new_text.text)
                replacements.append((idx, initial_result.attacked_text.words[idx]))

        if len(txts) == 0:
            return initial_adv_text

        similarity_scores = self._get_similarity_score(
            txts, initial_result.attacked_text.text
        )
        # Sort the replacements based on semantic similarity and replace
        # the substitute words with their original counterparts till the
        # remains adversarial.
        replacements = list(zip(similarity_scores, replacements))
        replacements.sort()
        replacements.reverse()

        for score, replacement in replacements:

            new_text = initial_adv_text.replace_word_at_index(
                replacement[0], replacement[1]
            )
            results, _ = self.get_goal_results([new_text])

            if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                initial_adv_text = new_text
            else:
                break
        return initial_adv_text

    def _crossover(self, text_1, text_2, initial_text, best_attack):
        """Does crossover step on `text_1` and `text_2` to yield a child text.

        Args:
            text_1 (AttackedText): Parent text 1.
            text_2 (AttackedText): Parent text 2.
            initial_text (AttackedText): The original text input.
            best_attack (PopulationMember): The best adversarial attack until now.
        Returns:
            pop_member as `PopulationMember`
        """
        index_to_replace = []
        words_to_replace = []

        for i in range(len(text_1.words)):

            if np.random.uniform() < 0.5:
                index_to_replace.append(i)
                words_to_replace.append(text_2.words[i])

        new_text = text_1.attacked_text.replace_words_at_indices(
            index_to_replace, words_to_replace
        )
        num_child_different_words = len(new_text.all_words_diff(initial_text))
        num_best_different_words = len(
            best_attack.attacked_text.all_words_diff(initial_text)
        )
        # If the generated child has more perturbation, return.
        if num_child_different_words > num_best_different_words:
            return PopulationMember(
                text_1,
                attributes={
                    "similarity_score": best_attack.attributes["similarity_score"],
                    "valid_adversarial_example": False,
                },
            )
        else:
            for idx in index_to_replace:
                if new_text.words[idx] == text_1.words[idx]:
                    continue
                previous_similarity_score = self._get_similarity_score(
                    [best_attack.attacked_text],
                    initial_text,
                    window_around_idx=idx,
                    use_window=True,
                )
                new_similarity_score = self._get_similarity_score(
                    [new_text], initial_text, window_around_idx=idx, use_window=True
                )
                # filter out child texts which do not improve semantic similarity
                # within a `self.window_size`.
                if new_similarity_score < previous_similarity_score:
                    return PopulationMember(
                        text_1,
                        attributes={
                            "similarity_score": new_similarity_score,
                            "valid_adversarial_example": False,
                        },
                    )

            similarity_score = self._get_similarity_score(
                [new_text.text], initial_text.text
            )
            return PopulationMember(
                new_text,
                attributes={
                    "similarity_score": similarity_score,
                    "valid_adversarial_example": True,
                },
            )

    def _perform_search(self, initial_result):
        """Performs random Initialisation, Search Space Reduction and genetic
        Optimization to generate attack by observing only the topmost label.

        Args:
            initial_result (GoalFunctionResult): The result of the original input text`.
        Returns:
            best_result as `GoalFunctionResult`
        """
        # Random Initialization
        initial_adv_text, init_adv_result = self._generate_initial_adversarial_example(initial_result)

        if initial_result.attacked_text.words == initial_adv_text.words:
            return initial_result
        # Search space reduction
        new_text = self._search_space_reduction(initial_result, initial_adv_text)
        results, search_over = self.get_goal_results([new_text])
        if not search_over:
            return init_adv_result
    
        if results[0].goal_status != GoalFunctionResultStatus.SUCCEEDED:
            return results[0]

        initial_text = initial_result.attacked_text
        changed_indices = initial_text.all_words_diff(new_text)
        changed_indices = list(changed_indices)
        self._search_over = False
        # Initialize Population
        population = self._initialize_population(
            new_text, initial_text, changed_indices
        )
        if len(population) == 0:
            return results[0]

        similarity_score = self._get_similarity_score(
            [initial_adv_text.text], initial_text.text
        )[0]
        best_attack = PopulationMember(
            new_text,
            attributes={
                "similarity_score": similarity_score,
                "valid_adversarial_example": True,
            },
        )
        best_result, _ = self.get_goal_results([best_attack.attacked_text])
        best_result = best_result[0]

        max_replacements_left = defaultdict(int)

        for _ in range(self.max_iters):

            num_min_different_words = len(
                best_attack.attacked_text.all_words_diff(initial_text)
            )

            similarity_scores = []

            for pop_member in population:
                num_different_words = len(
                    pop_member.attacked_text.all_words_diff(initial_text)
                )
                similarity_scores.append(pop_member.attributes["similarity_score"])

                if num_different_words < num_min_different_words:
                    best_attack = pop_member
                    num_min_different_words = num_different_words

            best_result, _ = self.get_goal_results([best_attack.attacked_text])
            best_result = best_result[0]

            assert best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED

            similarity_scores = np.asarray(similarity_scores)
            similarity_scores = softmax(similarity_scores)
            # Selection
            parent1_idx = np.random.choice(
                len(population), size=self.pop_size - 1, p=similarity_scores
            )
            parent2_idx = np.random.choice(
                len(population), size=self.pop_size - 1, p=similarity_scores
            )

            childs = []
            child_texts = []
            final_childs = []
            # Crossover
            for i in range(self.pop_size - 1):

                new_child = self._crossover(
                    population[parent1_idx[i]],
                    population[parent2_idx[i]],
                    initial_text,
                    best_attack,
                )
                if new_child.attributes["valid_adversarial_example"]:
                    childs.append(new_child)
                    child_texts.append(new_child.attacked_text)

            results, _ = self.get_goal_results(child_texts)

            for i in range(len(results)):

                if results[i].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    final_childs.append((childs[i], results[i]))
            # In the original implementation of the paper the changed indices are not
            # updated in every iteration. Here, the changed indices are updated.
            changed_indices = initial_text.all_words_diff(best_attack.attacked_text)
            changed_indices = list(changed_indices)
            index_positions_to_perturb = np.random.choice(
                len(changed_indices), size=min(len(changed_indices), len(final_childs))
            )
            population = []
            # Mutation of children
            for i in range(len(index_positions_to_perturb)):

                child = final_childs[i][0]
                j = index_positions_to_perturb[i]

                if (
                    initial_text.words[changed_indices[j]]
                    == child.attacked_text.words[changed_indices[j]]
                ):
                    population.append(child)
                    continue

                if (
                    max_replacements_left[changed_indices[j]]
                    <= self.max_replacements_per_index
                ):
                    new_pop_member = self._perturb(
                        child,
                        final_childs[i][1],
                        initial_text,
                        changed_indices[j],
                        best_attack,
                    )
                    if new_pop_member.attributes["valid_adversarial_example"]:
                        population.append(new_pop_member)
                        max_replacements_left[changed_indices[j]] += 1

            population.append(best_attack)

        return best_result

    def check_transformation_compatibility(self, transformation):
        """The genetic algorithm is specifically designed for word
        substitutions."""
        return transformation_consists_of_word_swaps(transformation)

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["pop_size", "max_iters", "max_replacements_per_index"]

