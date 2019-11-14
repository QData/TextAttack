'''
Algorithm from Generating Natural Language Adversarial Examples by Alzantot et. al

`<arxiv.org/abs/1804.07998>`_

`<github.com/nesl/nlp_adversarial_examples>`_
'''

import numpy as np

from textattack.attacks import AttackResult, FailedAttackResult
from textattack.attacks.blackbox import BlackBoxAttack
from textattack.transformations import WordSwap
from copy import deepcopy

class GeneticAlgorithm(BlackBoxAttack):
    '''
    Attacks a model using a genetic algorithm. 

    Args:
        model: A PyTorch or TensorFlow model to attack.
        transformation: The type of transformation to use. Should be a subclass of WordSwap. 
        pop_size (:obj:`int`, optional): The population size. Defauls to 20. 
        max_iters (:obj:`int`, optional): The maximum number of iterations to use. Defaults to 50. 

    Raises:
        ValueError: If the transformation is not a subclass of WordSwap. 

    '''
    def __init__(self, model, transformation, pop_size=20, max_iters=50):
        if not isinstance(transformation, WordSwap):
            raise ValueError(f'Transformation is of type {type(transformation)}, should be a subclass of WordSwap')
        super().__init__(model)
        self.model = model
        self.transformation = transformation
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.temp = 0.3

    def _replace_at_index(self, pop_member, idx, original_label):
        """
        Select the best replacement for word at position (idx) 
        in (pop_member) to minimize score for original_label.

        Args:
            pop_member: The population member being perturbed.
            idx: The index at which to replace a word.
            original_label: The original prediction label.

        Returns:
            Whether a replacement which decreased the score was found.
        """
        transformations = self.get_transformations(self.transformation,
                                                   pop_member.tokenized_text,
                                                   original_text=self.original_tokenized_text,
                                                   indices_to_replace=[idx])
        if not len(transformations):
            return False
        new_x_preds = self._call_model(transformations)
        new_x_scores = new_x_preds[:, original_label]
        orig_score = self._call_model([pop_member.tokenized_text]).squeeze()[original_label]
        new_x_scores = orig_score - new_x_scores
        if new_x_scores.max() > 0:
            pop_member.tokenized_text = transformations[new_x_scores.argmax()]
            return True
        return False

    def _perturb(self, pop_member, original_label):
        '''
        Replaces a word in pop_member that has not been modified. 

        Args:
            pop_member: The population member being perturbed.
            original_label: The original prediction label.
        '''
        x_len = pop_member.neighbors_len.shape[0]
        neighbors_len = deepcopy(pop_member.neighbors_len)
        non_zero_indices = np.sum(np.sign(pop_member.neighbors_len))
        if non_zero_indices == 0:
            return
        iterations = 0
        while iterations < non_zero_indices:
            w_select_probs = neighbors_len / np.sum(neighbors_len)
            rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
            if self._replace_at_index(pop_member, rand_idx, original_label):
                pop_member.neighbors_len[rand_idx] = 0
                break
            neighbors_len[rand_idx] = 0
            iterations += 1

    def _generate_population(self, neighbors_len, original_label):
        '''
        Generates a population of texts each with one word replaced

        Args:
            neighbors_len: A list of the number of candidate neighbors for each word.
            original_label: The original prediction label.

        Returns:
            The population.
        '''
        pop = []
        for _ in range(self.pop_size):
            pop_member = PopulationMember(self.original_tokenized_text, deepcopy(neighbors_len))
            self._perturb(pop_member, original_label)
            pop.append(pop_member)
        return pop

    def _crossover(self, pop_member1, pop_member2):
        '''
        Generates a crossover between pop_member1 and pop_member2.

        Args:
            pop_member1: The first population member.
            pop_member2: The second population member.

        Returns:
            A population member containing the crossover.
        '''
        indices_to_replace = []
        words_to_replace = []
        x1_text = pop_member1.tokenized_text
        x2_words = pop_member2.tokenized_text.words()
        new_neighbors_len = deepcopy(pop_member1.neighbors_len)
        for i in range(len(x1_text.words())):
            if np.random.uniform() < 0.5:
                indices_to_replace.append(i)
                words_to_replace.append(x2_words[i])
                new_neighbors_len[i] = pop_member2.neighbors_len[i]
        new_text = x1_text.replace_words_at_indices(indices_to_replace, words_to_replace)
        return PopulationMember(new_text, deepcopy(new_neighbors_len))

    def _get_neighbors_len(self, tokenized_text):
        '''
        Generates this neighbors_len list

        Args:
            tokenized_text: The original text

        Returns:
            A list of number of candidate neighbors for each word
        '''
        words = tokenized_text.words()
        neighbors_list = [[] for _ in range(len(words))]
        transformations = self.get_transformations(self.transformation,
                                                   tokenized_text,
                                                   original_text=self.original_tokenized_text,
                                                   apply_constraints=False)
        for transformed_text in transformations:
            diff_idx = tokenized_text.first_word_diff_index(transformed_text)
            neighbors_list[diff_idx].append(transformed_text.words()[diff_idx])
        neighbors_list = [np.array(x) for x in neighbors_list]
        neighbors_len = np.array([len(x) for x in neighbors_list])
        return neighbors_len

    def _attack_one(self, original_label, tokenized_text):
        self.original_tokenized_text = tokenized_text
        neighbors_len = self._get_neighbors_len(tokenized_text)
        pop = self._generate_population(neighbors_len, original_label)
        for i in range(self.max_iters):
            pop_preds = self._call_model([pm.tokenized_text for pm in pop])
            pop_scores = pop_preds[:, original_label]
            print('\t\t', i, ' -- ', pop_scores.min())
            top_attack = pop_scores.argmin()

            logits = ((-pop_scores) / self.temp).exp()
            select_probs = (logits / logits.sum()).cpu().numpy()
            
            top_attack_probs = pop_preds[top_attack, :].cpu()
            if np.argmax(top_attack_probs) != original_label:
                return AttackResult(
                    self.original_tokenized_text,
                    pop[top_attack].tokenized_text,
                    original_label,
                    np.argmax(top_attack_probs)
                )

            elite = [pop[top_attack]]  # elite
            parent1_idx = np.random.choice(
                self.pop_size, size=self.pop_size-1, p=select_probs)
            parent2_idx = np.random.choice(
                self.pop_size, size=self.pop_size-1, p=select_probs)

            children = [self._crossover(pop[parent1_idx[i]], pop[parent2_idx[i]])
                                for i in range(self.pop_size-1)]
            for c in children:
                self._perturb(c, original_label)

            pop = elite + children

        return FailedAttackResult(
            self.original_tokenized_text,
            original_label
        )
    
    def __str__(self):
        return "Genetic Algorithm"

class PopulationMember:
    '''
    A member of the population during the course of the genetic algorithm.

    Args:
        tokenized_text: The tokenized text of the population member.
        neighbors_len: A list of the number of candidate neighbors list for each word.
    '''
    def __init__(self, tokenized_text, neighbors_len):
        self.tokenized_text = tokenized_text
        self.neighbors_len = neighbors_len
