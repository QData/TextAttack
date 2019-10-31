'''
Algorithm from Generating Natural Language Adversarial Examples by Alzantot et. al
arxiv.org/abs/1804.07998
github.com/nesl/nlp_adversarial_examples
'''

import numpy as np

from textattack.attacks import Attack, AttackResult
from textattack.transformations import WordSwap

class GeneticAlgorithm(Attack):
    def __init__(self, model, transformation, pop_size=20, max_iters=100, n1=20):
        if not isinstance(transformation, WordSwap):
            raise ValueError(f'Transformation is of type {type(transformation)}, should be a subclass of WordSwap')
        super().__init__(model)
        self.model = model
        self.transformation = transformation
        # self.batch_model = batch_model
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.top_n = n1  # similar words
        self.temp = 0.3

    def select_best_replacement(self, pos, x_cur, x_orig, target, replace_list):
        """ Select the most effective replacement to word at pos (pos)
            in (x_cur) between the words in replace_list.
        """
        orig_words = x_orig.words()
        new_x_list = [x_cur.replace_word_at_index(
            pos, w) if orig_words[pos] != w and w != '0.0' else x_cur for w in replace_list]
        new_x_preds = self._call_model(new_x_list)

        new_x_scores = new_x_preds[:, target]
        orig_score = self._call_model([x_cur]).squeeze()[target]
        new_x_scores = new_x_scores - orig_score

        # Eliminate words that are not that close
        new_x_scores[self.top_n:] = -10000000

        '''
        if self.use_lm:
            prefix = ""
            suffix = None
            if pos > 0:
                prefix = self.dataset.inv_dict[x_cur[pos-1]]
            #
            orig_word = self.dataset.inv_dict[x_orig[pos]]
            if self.use_suffix and pos < x_cur.shape[0]-1:
                if (x_cur[pos+1] != 0):
                    suffix = self.dataset.inv_dict[x_cur[pos+1]]
            # print('** ', orig_word)
            replace_words_and_orig = [
                self.dataset.inv_dict[w] if w in self.dataset.inv_dict else 'UNK' for w in replace_list[:self.top_n]] + [orig_word]
            # print(replace_words_and_orig)
            replace_words_lm_scores = self.lm.get_words_probs(
                prefix, replace_words_and_orig, suffix)
            # print(replace_words_lm_scores)
            # for i in range(len(replace_words_and_orig)):
            #    print(replace_words_and_orig[i], ' -- ', replace_words_lm_scores[i])

            # select words
            new_words_lm_scores = np.array(replace_words_lm_scores[:-1])
            # abs_diff_lm_scores = np.abs(new_words_lm_scores - replace_words_lm_scores[-1])
            # rank_replaces_by_lm = np.argsort(abs_diff_lm_scores)
            rank_replaces_by_lm = np.argsort(-new_words_lm_scores)

            filtered_words_idx = rank_replaces_by_lm[self.top_n2:]
            # print(filtered_words_idx)
            new_x_scores[filtered_words_idx] = -10000000
        '''

        if (new_x_scores.max() > 0):
            return new_x_list[new_x_scores.argmax()]

        return x_cur

    def perturb(self, x_cur, x_orig, neighbors, w_select_probs, target):
        # Pick a word that is not modified and is not UNK
        x_len = w_select_probs.shape[0]
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        diff_set = x_cur.all_words_diff(x_orig)
        num_replaceable_words = np.sum(np.sign(w_select_probs))
        while len(diff_set) < num_replaceable_words and x_cur.ith_word_diff(x_orig, rand_idx):
            ''' The condition above has a quick hack to prevent getting stuck in infinite loop while processing too short examples and all words (excluding articles) have been already replaced and still no successful attack is found. A more elegant way to handle this could be implemented in Attack to abort early based on the status of all population members or to improve select_best_replacement by making it stochastic.
            '''
            rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]

        replace_list = neighbors[rand_idx]
        if len(replace_list) < self.top_n:
            replace_list = np.concatenate(
                (replace_list, np.zeros(self.top_n - replace_list.shape[0])))
        return self.select_best_replacement(rand_idx, x_cur, x_orig, target, replace_list)

    def generate_population(self, x_orig, neigbhors_list, w_select_probs, target, pop_size):
        return [self.perturb(x_orig, x_orig, neigbhors_list, w_select_probs, target) for _ in range(pop_size)]

    def crossover(self, x1, x2):
        indices_to_replace = []
        words_to_replace = []
        x2_words = x2.words()
        for i in range(len(x1.words())):
            if np.random.uniform() < 0.5:
                indices_to_replace.append(i)
                words_to_replace.append(x2_words[i])
        return x1.replace_words_at_indices(indices_to_replace, words_to_replace)

    def _get_neighbors(self, tokenized_text, original_tokenized_text):
        words = tokenized_text.words()
        neighbors_list = [[] for _ in range(len(words))]
        transformations = self.get_transformations(self.transformation,
                                                   tokenized_text,
                                                   original_text=original_tokenized_text)
        diff_idx = 0
        for transformed_text in transformations:
            while not tokenized_text.ith_word_diff(transformed_text, diff_idx):
                diff_idx += 1
            neighbors_list[diff_idx].append(transformed_text.words()[diff_idx])
        neighbors_list = [np.array(x) for x in neighbors_list]
        neighbors_len = [len(x) for x in neighbors_list]
        w_select_probs = neighbors_len / np.sum(neighbors_len)
        return neighbors_list, w_select_probs 

    def _attack_one(self, original_label, tokenized_text):
        target = 1 - original_label
        original_tokenized_text = tokenized_text
        neighbors_list, w_select_probs = self._get_neighbors(
            tokenized_text, original_tokenized_text)
        pop = self.generate_population(
            original_tokenized_text, neighbors_list, w_select_probs, target, self.pop_size)
        for i in range(self.max_iters):
            pop_preds = self._call_model(pop)
            pop_scores = pop_preds[:, target]
            print('\t\t', i, ' -- ', pop_scores.max())
            top_attack = pop_scores.argmax()

            logits = (pop_scores / self.temp).exp()
            select_probs = (logits / logits.sum()).cpu().numpy()
            
            top_attack_probs = pop_preds[top_attack, :].cpu()
            if np.argmax(top_attack_probs) == target:
                return AttackResult(
                    original_tokenized_text,
                    pop[top_attack],
                    original_label,
                    target
                )

            elite = [pop[top_attack]]  # elite
            # print(select_probs.shape)
            parent1_idx = np.random.choice(
                self.pop_size, size=self.pop_size-1, p=select_probs)
            parent2_idx = np.random.choice(
                self.pop_size, size=self.pop_size-1, p=select_probs)

            initial_children = [self.crossover(pop[parent1_idx[i]],
                                     pop[parent2_idx[i]])
                      for i in range(self.pop_size-1)]
            children = []

            for child in initial_children:
                neighbors_list, w_select_probs = self._get_neighbors(
                    child, original_tokenized_text)
                children.append(self.perturb(
                    child, original_tokenized_text, neighbors_list, w_select_probs, target))

            pop = elite + children

        return AttackResult(
            original_tokenized_text,
            pop[top_attack],
            original_label,
            target
        )
    

